# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertForTokenClassification
from contrastive import instance_CL_Loss, proto_CL_Loss

import pdb
logger = logging.getLogger(__file__)


class BertForTokenClassification_(BertForTokenClassification):
    def __init__(self, *args, **kwargs):
        super(BertForTokenClassification_, self).__init__(*args, **kwargs)
        self.input_size = 768
        self.span_loss = nn.functional.cross_entropy
        self.type_loss = nn.functional.cross_entropy
        self.dropout = nn.Dropout(p=0.1)
        self.log_softmax = nn.functional.log_softmax

    def set_config(
        self,
        use_classify: bool = False,
        distance_mode: str = "cos",
        similar_k: float = 30,
        shared_bert: bool = True,
        train_mode: str = "add",
        context_proto: bool = False, #自己加的变量
        mask_proto: bool = False, #自己加的变量
        instance_mask_cL_Loss: bool = False, #自己加的变量
        instance_context_cL_Loss: bool = False, #自己加的变量
        proto_cL_Loss: bool = False, #自己加的变量
    ):
        self.use_classify = use_classify
        self.distance_mode = distance_mode
        self.similar_k = similar_k
        self.shared_bert = shared_bert
        self.train_mode = train_mode
        self.context_proto = context_proto
        self.mask_proto = mask_proto
        self.instance_mask_cL_Loss = instance_mask_cL_Loss
        self.instance_context_cL_Loss = instance_context_cL_Loss
        self.proto_cL_Loss = proto_cL_Loss


        if train_mode == "type":
            self.classifier = None

        if train_mode != "span":
            self.ln = nn.LayerNorm(768, 1e-5, True)
            if use_classify:
                # self.type_classify = nn.Linear(self.input_size, self.input_size)
                self.type_classify = nn.Sequential(
                    nn.Linear(self.input_size, self.input_size * 2),
                    nn.GELU(),
                    nn.Linear(self.input_size * 2, self.input_size),
                )
            if self.distance_mode != "cos":
                self.dis_cls = nn.Sequential(
                    nn.Linear(self.input_size * 3, self.input_size),
                    nn.GELU(),
                    nn.Linear(self.input_size, 2),
                )
        config = {
            "use_classify": use_classify,
            "distance_mode": distance_mode,
            "similar_k": similar_k,
            "shared_bert": shared_bert,
            "train_mode": train_mode,
            "context_proto": context_proto,
            "mask_proto": mask_proto,
            "instance_mask_cL_Loss":instance_mask_cL_Loss,
            "instance_context_cL_Loss":instance_context_cL_Loss,
            "proto_cL_Loss":proto_cL_Loss,

        }
        logger.info(f"Model Setting: {config}")
        if not shared_bert:
            self.bert2 = deepcopy(self.bert)
    

    def forward_wuqh(
        self,
        input_ids,
        # context_ids, #mask 实体的ids
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        e_mask=None,
        e_type_ids=None,
        e_type_mask=None,
        entity_types=None,
        entity_mode: str = "mean",
        is_update_type_embedding: bool = False,
        lambda_max_loss: float = 0.0,
        sim_k: float = 0,
    ):      
        
        max_len = (attention_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len].type(torch.int8)
        token_type_ids = token_type_ids[:, :max_len]
        labels = labels[:, :max_len]
        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )


        sequence_output = self.dropout(output[0])

 

        if self.train_mode != "type":
            logits = self.classifier(
                sequence_output
            )  # batch_size x seq_len x num_labels
        else:
            logits = None
        if not self.shared_bert:
            output2 = self.bert2(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
            sequence_output2 = self.dropout(output2[0])

        if e_type_ids is not None and self.train_mode != "span":  #type分类走这个

            if e_type_mask.sum() != 0:
                M = (e_type_mask[:, :, 0] != 0).max(0)[0].nonzero(as_tuple=False)[
                    -1
                ].item() + 1
            else:
                M = 1
            e_mask = e_mask[:, :M, :max_len].type(torch.int8)
            e_type_ids = e_type_ids[:, :M, :]
            e_type_mask = e_type_mask[:, :M, :].type(torch.int8)
            B, M, K = e_type_ids.shape

            '''1. 原代码句向量表征（e_out）：构建每个实体的句向量表征'''
            e_out = self.get_enity_hidden(
                sequence_output if self.shared_bert else sequence_output2,
                e_mask,
                entity_mode,
            )  #把max sequence length平均掉，max entity num代表每句话的不同实体数量, 所以这里得到的句向量
            if self.use_classify:
                e_out = self.type_classify(e_out)
            e_out = self.ln(e_out)  #归一化 batch_size x max_entity_num x hidden_size 这其实就是原型的隐藏向量
            '''原代码流程：构建e_out的原型，维护一个type embedding，最后通过与原型的cos度量算loss'''
            if is_update_type_embedding:
                entity_types.update_type_embedding(e_out, e_type_ids, e_type_mask, type='origin') #label的原型type embedding weight更新了
            #原本的原型：保存type_embedding变量中，可以再entity_types类中设置是否更新这个embedding
            # TODO 下面两行，检查是否合理
            e_out = e_out.unsqueeze(2).expand(B, M, K, -1) 
            types = self.get_types_embedding(  
                e_type_ids, entity_types
            )  # batch_size x max_entity_num x K x hidden_size   embedding位置
            if self.distance_mode == "cat":
                e_types = torch.cat([e_out, types, (e_out - types).abs()], -1)
                e_types = self.dis_cls(e_types)
                e_types = e_types[:, :, :0]
            elif self.distance_mode == "l2":
                e_types = -(torch.pow(e_out - types, 2)).sum(-1)
            elif self.distance_mode == "cos":
                sim_k = sim_k if sim_k else self.similar_k
                e_types = sim_k * (e_out * types).sum(-1) / 768   # batch_size x max_entity_num x K, 隐层被降维了
            e_logits = e_types
            if M:
                # TODO 检查合理性，检查是不是写错了
                em = e_type_mask.clone()
                em[em.sum(-1) == 0] = 1
                e = e_types * em
                e_type_label = torch.zeros((B, M)).to(e_types.device)
                type_loss = self.calc_loss(
                    self.type_loss, e, e_type_label, e_type_mask[:, :, 0]
                )
            # else:
            #     type_loss = torch.tensor(0).to(sequence_output.device) #这句先注释掉，目前不走
            '''原proto的对比学习，正样本为同类别的原型，负样本为非同类别的原型(也可以改为原样本的其余实体，proto_CL_Loss中origin，aug模式分别对应两种负样本模式)'''
            if self.proto_cL_Loss:
                e_hidden, e_labels=self.set_hidden(e_out[:, :, 0, :], e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据
                proto_origin = self.set_proto(e_labels,entity_types)
                proto_cl_loss = proto_CL_Loss(e_hidden,proto_origin)
                type_loss = 0.2 * proto_cl_loss + type_loss


            if self.mask_proto:
                '''2. MASK句向量表征（e_out_mask）：将当前句子中无关实体用mask表示后过bert'''
                # 获取输入BERT的MASK掉其余实体的输入
                mask_ids, attention_mask_mask, token_type_ids_mask, e_mask_masked, e_type_ids_masked = self.get_inputs_for_masked_type(input_ids, e_mask, attention_mask, e_type_ids)
                output_masked = self.bert(mask_ids, attention_mask=attention_mask_mask, token_type_ids=token_type_ids_mask)
                sequence_output_masked = self.dropout(output_masked[0])
                e_out_masked = self.get_enity_hidden(sequence_output_masked, e_mask_masked, entity_mode)
                if self.use_classify:
                    e_out_masked = self.type_classify(e_out_masked)
                e_out_masked = self.ln(e_out_masked)  #e_out_mask是每个实体的句向量的表征，在算每个实体表征时均mask其余实体，维度：batch_size x max_entity_num x hidden_size
                e_type_mask_masked = torch.ones(mask_ids.shape[0], dtype=torch.int64).reshape(mask_ids.shape[0], 1, 1).to(mask_ids.device)
                if is_update_type_embedding:
                    entity_types.update_type_embedding(e_out_masked, e_type_ids_masked, e_type_mask_masked, type='masked') #label的原型type embedding weight更新了
                #mask其余实体的原型：保存type_mask_embedding变量中，可以再entity_types类中设置是否更新这个embedding
                e_out_masked = e_out_masked.unsqueeze(2).expand(*e_type_ids_masked.shape, -1) 
                types_masked = self.get_types_mask_embedding(  
                    e_type_ids_masked, entity_types
                )  # batch_size x max_entity_num x K x hidden_size   embedding位置
                if self.distance_mode == "cat":
                    e_types = torch.cat([e_out_masked, types_masked, (e_out - types_masked).abs()], -1)
                    e_types = self.dis_cls(e_types)
                    e_types = e_types[:, :, :0]
                elif self.distance_mode == "l2":
                    e_types = -(torch.pow(e_out_masked - types_masked, 2)).sum(-1)
                elif self.distance_mode == "cos":
                    sim_k = sim_k if sim_k else self.similar_k
                    e_types = sim_k * (e_out_masked * types_masked).sum(-1) / 768

                e_logits_mask = e_types
                if M:
                    em = e_type_mask_masked.clone()
                    em[em.sum(-1) == 0] = 1
                    e = e_types * em
                    e_type_label = torch.zeros(*e_type_ids_masked.shape[:2]).to(e_types.device)
                    type_loss_masked = self.calc_loss(
                        self.type_loss, e, e_type_label, e_type_mask_masked[:, :, 0])
                    type_loss=type_loss+0.2*type_loss_masked #直接通过mask原型计算loss，不加可以注释掉
                
                if self.proto_cL_Loss: #加入原型对比学习
                    #这里可以尝试一下e_hidden_mask为原样本的效果
                    proto_origin = self.set_proto_mask(e_labels,entity_types)
                    proto_cl_loss = proto_CL_Loss(e_hidden,proto_origin)
                    type_loss = 0.2 * proto_cl_loss + type_loss #mask原型的cl loss                    


            # TODO 跑通加context原型的代码
            if self.context_proto:
                '''3. 构建上下文的句向量表征（e_out_context）：mask所有的实体，只保留句向量'''
                e_mask_context = e_mask.clone()  #把值给clone过来,这里只反转了e_mask
                e_mask_context = 1 - e_mask_context
                
                e_out_context = self.get_enity_hidden( 
                    sequence_output if self.shared_bert else sequence_output2,
                    e_mask_context, #这个矩阵实体的位置为0，非实体为1，与sequence_output相乘后只剩下上下文的表征
                    entity_mode,
                )  
                if self.use_classify:
                    e_out_context = self.type_classify(e_out_context)
                e_out_context = self.ln(e_out_context)
                if is_update_type_embedding:
                    entity_types.update_type_embedding(e_out_context, e_type_ids, e_type_mask, type='context') #label的原型type embedding weight更新了
                
                #上下文的原型：保存type_context_embedding变量中，可以再entity_types类中设置是否更新这个embedding
                e_out_context = e_out_context.unsqueeze(2).expand(B, M, K, -1) 
                types_context = self.get_types_context_embedding(  
                    e_type_ids, entity_types
                )  # batch_size x max_entity_num x K x hidden_size   embedding位置
                if self.distance_mode == "cat":
                    e_types = torch.cat([e_out, types, (e_out - types).abs()], -1)
                    e_types = self.dis_cls(e_types)
                    e_types = e_types[:, :, :0]
                elif self.distance_mode == "l2":
                    e_types = -(torch.pow(e_out_context - types_context, 2)).sum(-1)
                elif self.distance_mode == "cos":
                    sim_k = sim_k if sim_k else self.similar_k
                    e_types = sim_k * (e_out_context * types_context).sum(-1) / 768
                e_logits_context = e_types
                if M:
                    em = e_type_mask.clone()
                    em[em.sum(-1) == 0] = 1
                    e = e_types * em
                    e_type_label = torch.zeros((B, M)).to(e_types.device)
                    type_loss_context = self.calc_loss(
                        self.type_loss, e, e_type_label, e_type_mask[:, :, 0]
                    )
                    type_loss=type_loss+0.2*type_loss_context #如果更新原型度量直接计算的loss，则加这句，否则可以注释掉

                if self.proto_cL_Loss: 
                    # e_hidden, e_labels=self.set_hidden(e_out, e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据             
                    proto_origin = self.set_proto_context(e_labels,entity_types)
                    proto_cl_loss = proto_CL_Loss(e_hidden,proto_origin)
                    type_loss = 0.2 * proto_cl_loss + type_loss #上下文原型cl loss



            # TODO: 1. 重命名变量；2. 重命名损失变量；3. 调整代码顺序；4. 检查更新masked 和 context类原型是否正确
            # TODO 重构loss的和；用多任务调点   
            '''2个instance-instance级别的对比学习'''
            if self.instance_mask_cL_Loss:
                '''mask其他实体为正样本的对比学习，负样本可以为原样本的其他样例'''
                e_hidden, e_labels=self.set_hidden(e_out, e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据
                e_mask_hidden, e_mask_labels=self.set_hidden(e_out_masked, e_type_ids, e_type_mask, entity_types)
                instance_mask_cL_Loss = instance_CL_Loss(e_hidden,e_mask_hidden) #以mask其他实体为正样本的对比学习，负样本可以为原样本的其他样例，也可以正样本的其余样本
                type_loss = 0.2 * instance_mask_cL_Loss + type_loss

                '''以上下文为正样本的对比学习，负样本可以为原样本的其他样例，也可以正样本的其余样本'''
                e_hidden, e_labels=self.set_hidden(e_out, e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据
                e_context_hidden, e_context_labels=self.set_hidden(e_out_context, e_type_ids, e_type_mask, entity_types)
                instance_context_cL_Loss = instance_CL_Loss(e_hidden,e_context_hidden) #以上下文为正样本的对比学习，负样本可以为原样本的其他样例，也可以正样本的其余样本
                type_loss = 0.2 * instance_context_cL_Loss + type_loss

            
            
                

        if labels is not None and self.train_mode != "type":   ###span检测走这个
            # Only keep active parts of the loss
            loss_fct = CrossEntropyLoss(reduction="none")
            B, M, T = logits.shape
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.reshape(-1, self.num_labels)[active_loss]
                active_labels = labels.reshape(-1)[active_loss]
                base_loss = loss_fct(active_logits, active_labels)
                loss = torch.mean(base_loss)

                # max-loss
                if lambda_max_loss > 0:
                    active_loss = active_loss.view(B, M)
                    active_max = []
                    start_id = 0
                    for i in range(B):
                        sent_len = torch.sum(active_loss[i])
                        end_id = start_id + sent_len
                        active_max.append(torch.max(base_loss[start_id:end_id]))
                        start_id = end_id

                    loss += lambda_max_loss * torch.mean(torch.stack(active_max))
            
            else:
                raise ValueError("Miss attention mask!")
        else:
            loss = None

        return logits, e_logits, loss, type_loss
    
    def get_inputs_for_masked_type(self, input_ids, e_mask, attention_mask_ori, e_type_ids_ori):
        device = input_ids.device
        B, M, T = e_mask.shape
        mask_ids = []
        attention_mask_masked = []
        e_mask_masked = []
        e_type_ids_masked = []
        for i in range(B):
            sum_e_mask = torch.sum(e_mask[i].to('cpu')*torch.arange(1, 1+M).reshape(M, 1), dim=0)
            for j in range(1, 1+torch.max(sum_e_mask)):
                # 除了j和0都要mask掉
                input_ids_clone = input_ids[i].to('cpu')
                for k in range(T):
                    if sum_e_mask[k] not in [0, j]:
                        input_ids_clone[k] = 103
                mask_ids.append(input_ids_clone)
                attention_mask_masked.append(attention_mask_ori[i])
                e_mask_masked.append(e_mask[i][j-1])
                e_type_ids_masked.append(e_type_ids_ori[i][j-1])


        mask_ids = torch.stack(mask_ids).to(device)
        attention_mask_masked = torch.stack(attention_mask_masked)
        token_type_ids_masked = torch.zeros(mask_ids.shape[0], T, dtype=torch.int64, device=device)
        e_mask_masked = torch.stack(e_mask_masked).unsqueeze(1)
        e_type_ids_masked = torch.stack(e_type_ids_masked).unsqueeze(1)

        return mask_ids, attention_mask_masked, token_type_ids_masked, e_mask_masked, e_type_ids_masked

    def get_enity_hidden( #最终得到batch size * max entity * hidden_size的句向量
        self, hidden: torch.Tensor, e_mask: torch.Tensor, entity_mode: str
    ):      
        B, M, T = e_mask.shape
        #hidden.unsqueeze(1).expand(B, M, T, -1)即在sequence前面加了一层max emtity，乘后项e_mask（只有实体词置1，其余为0的矩阵），最后得到
        #batch_size x max_entity_num x seq_len x hidden_size，其中seq_len只有实体词有向量，非实体词都为0.
        e_out = hidden.unsqueeze(1).expand(B, M, T, -1) * e_mask.unsqueeze(    
            -1
        )  # batch_size x max_entity_num x seq_len x hidden_size
        

        if entity_mode == "mean": #对seq_len那一维度加和，相当于现在e_out为每个句子实体词的表征
            return e_out.sum(2) / (e_mask.sum(-1).unsqueeze(-1) + 1e-30)  # batch_size x max_entity_num x hidden_size
    
    
    def set_hidden(self,e_out, e_type_ids, e_type_mask,entity_types):  #对比学习时数据预处理，去除重复类别的数据，返回去重的embedding，以及对应的原型
        hiddens = e_out[e_type_mask[:, :, 0] == 1]
        labels = e_type_ids[:, :, 0][e_type_mask[:, :, 0] == 1]
        set_labels = []  #去重的labels
        set_hiddens=[]   #去重的hiddens
        # hiddens,labels =list(hiddens),list(labels)
        slow = 0
        fast = 1
        while fast<len(labels): #滑动窗口，如果窗口两个相等，就放进去一个，不等则都放进去
            if labels[fast]==labels[slow]:
                set_labels.append(labels[slow]) #发现重复，则往前走一格，因为数据中只有最多两个重复的
                set_hiddens.append(hiddens[slow])
                slow+=1
                fast+=1
            else:
                set_labels.append(labels[slow])
                set_hiddens.append(hiddens[slow])
                
            slow+=1
            fast+=1
        
        #根据label去重复类别，最后返回的hiddens数据，均为不同实体的数据
        return torch.stack(set_hiddens), torch.stack(set_labels)

    #TODO 合并这些函数
    def set_proto(self, labels, entity_types): #得到对比学习所需要的原型,只取与之前方法对应的
        proto=[]
        for i in labels:
            proto.append(entity_types.types_embedding.weight.data[i])
        return torch.stack(proto)
    def set_proto_mask(self, labels, entity_types): #得到对比学习所需要的原型,只取与之前方法对应的
        proto=[]
        for i in labels:
            proto.append(entity_types.types_mask_embedding.weight.data[i])
        return torch.stack(proto)
    def set_proto_context(self, labels, entity_types): #得到对比学习所需要的原型,只取与之前方法对应的
        proto=[]
        for i in labels:
            proto.append(entity_types.types_context_embedding.weight.data[i])
        return torch.stack(proto)

    def get_types_embedding(self, e_type_ids: torch.Tensor, entity_types):
        return entity_types.get_types_embedding(e_type_ids)  #又调用了preprocessor里的types embedding。这里原型网络将参数更新了
    
    def get_types_mask_embedding(self, e_type_ids: torch.Tensor, entity_types): #获得类别mask的embedding
        return entity_types.get_types_mask_embedding(e_type_ids)  
    
    def get_types_context_embedding(self, e_type_ids: torch.Tensor, entity_types): #获得context的embedding
        return entity_types.get_types_context_embedding(e_type_ids)  

    def calc_loss(self, loss_fn, preds, target, mask=None):
        target = target.reshape(-1)
        preds += 1e-10
        preds = preds.reshape(-1, preds.shape[-1])
        ce_loss = loss_fn(preds, target.long(), reduction="none")
        if mask is not None:
            mask = mask.reshape(-1)
            ce_loss = ce_loss * mask
            return ce_loss.sum() / (mask.sum() + 1e-10)
        return ce_loss.sum() / (target.sum() + 1e-10)


class ViterbiDecoder(object):
    def __init__(
        self,
        id2label,
        transition_matrix,
        ignore_token_label_id=torch.nn.CrossEntropyLoss().ignore_index,
    ):
        self.id2label = id2label
        self.n_labels = len(id2label)
        self.transitions = transition_matrix
        self.ignore_token_label_id = ignore_token_label_id

    def forward(self, logprobs, attention_mask, label_ids):
        # probs: batch_size x max_seq_len x n_labels
        batch_size, max_seq_len, n_labels = logprobs.size()
        attention_mask = attention_mask[:, :max_seq_len]
        label_ids = label_ids[:, :max_seq_len]

        active_tokens = (attention_mask == 1) & (
            label_ids != self.ignore_token_label_id
        )
        if n_labels != self.n_labels:
            raise ValueError("Labels do not match!")

        label_seqs = []

        for idx in range(batch_size):
            logprob_i = logprobs[idx, :, :][
                active_tokens[idx]
            ]  # seq_len(active) x n_labels

            back_pointers = []

            forward_var = logprob_i[0]  # n_labels

            for j in range(1, len(logprob_i)):  # for tag_feat in feat:
                next_label_var = forward_var + self.transitions  # n_labels x n_labels
                viterbivars_t, bptrs_t = torch.max(next_label_var, dim=1)  # n_labels

                logp_j = logprob_i[j]  # n_labels
                forward_var = viterbivars_t + logp_j  # n_labels
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            path_score, best_label_id = torch.max(forward_var, dim=-1)
            best_label_id = best_label_id.item()
            best_path = [best_label_id]

            for bptrs_t in reversed(back_pointers):
                best_label_id = bptrs_t[best_label_id]
                best_path.append(best_label_id)

            if len(best_path) != len(logprob_i):
                raise ValueError("Number of labels doesn't match!")

            best_path.reverse()
            label_seqs.append(best_path)

        return label_seqs
