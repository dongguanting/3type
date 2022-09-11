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
            # import pdb
            # pdb.set_trace()

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
            # import pdb
            # pdb.set_trace()

            '''1. MASK句向量表征（e_out_mask）：将当前句子中无关实体用mask表示后过bert'''
            mask_ids= self.get_mask_ids(input_ids,e_mask,e_type_mask) #自己写的，获取除了目标实体外的实体，并将它mask掉，将用在get entity hidden函数
            mask_ids = mask_ids.reshape([B*M,max_len]) #[Batch size * Maxnums ,max_len]
            import pdb
            pdb.set_trace()
            attention_mask_mask = attention_mask.clone() #与mask ids 对齐维度
            token_type_ids_mask = token_type_ids.clone()
            attention_mask_mask=attention_mask_mask.repeat(1,M).reshape([B*M,max_len])
            token_type_ids_mask=token_type_ids_mask.repeat(1,M).reshape([B*M,max_len])
            sequence_output_mask_all=[]
            for i in range(0,M):
                output_mask = self.bert(mask_ids[B*i:B*(i+1),:], attention_mask=attention_mask_mask[B*i:B*(i+1),:], token_type_ids=token_type_ids_mask[B*i:B*(i+1),:]) #过bert取最后一维度的表征
                sequence_output_mask = self.dropout(output_mask[0]) #只能分两批送进去，然后合起来
                sequence_output_mask_all.append(sequence_output_mask)
            sequence_output_mask_all= torch.stack(sequence_output_mask_all)
            sequence_output_mask_all=sequence_output_mask_all.reshape([B*M,max_len,768]).reshape([B,M,max_len,768]) #sequence_output_mask_all：获得mask其余实体过bert的向量，维度[B, M, max_len, 768]
            
            e_out_mask = self.get_enity_hidden_mask(sequence_output_mask_all, e_mask, entity_mode,)
            if self.use_classify:
                e_out_mask = self.type_classify(e_out_mask)
            e_out_mask = self.ln(e_out_mask)  #e_out_mask是每个实体的句向量的表征，在算每个实体表征时均mask其余实体，维度：batch_size x max_entity_num x hidden_size
                
           
            '''2. 原代码句向量表征（e_out）：构建每个实体的句向量表征'''
            e_out = self.get_enity_hidden(
                sequence_output if self.shared_bert else sequence_output2,
                e_mask,
                entity_mode,
            )  #把max sequence length平均掉，max entity num代表每句话的不同实体数量, 所以这里得到的句向量
            e_out = self.ln(e_out)  #归一化 batch_size x max_entity_num x hidden_size 这其实就是原型的隐藏向量


            '''3. 构建上下文的句向量表征（e_out_context）：mask所有的实体，只保留句向量'''
            e_mask_context = e_mask.clone()  #把值给clone过来,这里只反转了e_mask
            e_mask_context[e_mask_context==1]=2
            e_mask_context[e_mask_context==0]=1
            e_mask_context[e_mask_context==2]=0
            
            e_out_context = self.get_enity_hidden( 
                sequence_output if self.shared_bert else sequence_output2,
                e_mask_context, #这个矩阵实体的位置为0，非实体为1，与sequence_output相乘后只剩下上下文的表征
                entity_mode,
            )  
            e_out_context = self.ln(e_out_context)

            if self.use_classify:
                e_out = self.type_classify(e_out)
            
            '''原代码流程：构建e_out的原型，维护一个type embedding，最后通过与原型的cos度量算loss'''
            if is_update_type_embedding:
                entity_types.update_type_embedding(e_out, e_type_ids, e_type_mask) #label的原型type embedding weight更新了
            #原本的原型：保存type_embedding变量中，可以再entity_types类中设置是否更新这个embedding
           
           
            '''2个instance-instance级别的对比学习'''
            if self.instance_mask_cL_Loss:
                '''mask其他实体为正样本的对比学习，负样本可以为原样本的其他样例'''
                e_hidden, e_labels=self.set_hidden(e_out, e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据
                e_mask_hidden, e_mask_labels=self.set_hidden(e_out_mask, e_type_ids, e_type_mask, entity_types)
                

                instance_mask_cL_Loss = instance_CL_Loss(e_hidden,e_mask_hidden) #以mask其他实体为正样本的对比学习，负样本可以为原样本的其他样例，也可以正样本的其余样本
            
            if self.instance_context_cL_Loss:
                '''以上下文为正样本的对比学习，负样本可以为原样本的其他样例，也可以正样本的其余样本'''
                e_hidden, e_labels=self.set_hidden(e_out, e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据
                e_context_hidden, e_context_labels=self.set_hidden(e_out_context, e_type_ids, e_type_mask, entity_types)
                instance_context_cL_Loss = instance_CL_Loss(e_hidden,e_context_hidden) #以上下文为正样本的对比学习，负样本可以为原样本的其他样例，也可以正样本的其余样本
            
            
            '''原proto的对比学习，正样本为同类别的原型，负样本为非同类别的原型(也可以改为原样本的其余实体，proto_CL_Loss中origin，aug模式分别对应两种负样本模式)'''
            if self.proto_cL_Loss:
                e_hidden, e_labels=self.set_hidden(e_out, e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据
               
                proto_origin = self.set_proto(e_labels,entity_types)
                proto_cl_loss = proto_CL_Loss(e_hidden,proto_origin)
            
            # import pdb
            # pdb.set_trace()
            

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
                em = e_type_mask.clone()
                em[em.sum(-1) == 0] = 1
                e = e_types * em
                e_type_label = torch.zeros((B, M)).to(e_types.device)
                type_loss = self.calc_loss(
                    self.type_loss, e, e_type_label, e_type_mask[:, :, 0]
                )
                if self.instance_mask_cL_Loss: #加入对比学习的两个实例级loss
                    type_loss = 0.2 * instance_mask_cL_Loss + type_loss
                if self.instance_context_cL_Loss:   
                    type_loss = 0.2 * instance_context_cL_Loss + type_loss
                if self.proto_cL_Loss:  #加入原本原型的cl-loss 
                    type_loss = 0.2 * proto_cl_loss + type_loss
                

            '''计算上下文的原型'''
            if self.context_proto:
                #上下文的原型：保存type_context_embedding变量中，可以再entity_types类中设置是否更新这个embedding
                if is_update_type_embedding:
                    entity_types.update_type_context_embedding(e_out_context, e_type_ids, e_type_mask) #label的原型type embedding weight更新了
                e_out_context = e_out_context.unsqueeze(2).expand(B, M, K, -1) 
                types_context = self.get_types_context_embedding(  
                    e_type_ids, entity_types
                )  # batch_size x max_entity_num x K x hidden_size   embedding位置
                '''上下文原型的原型对比学习'''
                if self.proto_cL_Loss: 
                    # e_hidden, e_labels=self.set_hidden(e_out, e_type_ids, e_type_mask, entity_types) #因为数据中有可能同一类别出现若干条，我把它们去重，每种类别保留一条，这样符合batch内负样本均为不同类别数据             
                    proto_origin = self.set_proto_context(e_labels,entity_types)
                    proto_cl_loss = proto_CL_Loss(e_hidden,proto_origin)

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
                        type_loss = 0.2 * proto_cl_loss + type_loss #上下文原型cl loss

            '''计算mask原型'''
            if self.mask_proto:
                
                
                #mask其余实体的原型：保存type_mask_embedding变量中，可以再entity_types类中设置是否更新这个embedding
                if is_update_type_embedding:
                    entity_types.update_type_mask_embedding(e_out_mask, e_type_ids, e_type_mask) #label的原型type embedding weight更新了
                e_out_mask = e_out_mask.unsqueeze(2).expand(B, M, K, -1) 
                types_mask = self.get_types_mask_embedding(  
                    e_type_ids, entity_types
                )  # batch_size x max_entity_num x K x hidden_size   embedding位置

                if self.proto_cL_Loss: #加入原型对比学习
                    
                    #这里可以尝试一下e_hidden_mask为原样本的效果
                    proto_origin = self.set_proto_mask(e_labels,entity_types)
                    proto_cl_loss = proto_CL_Loss(e_hidden,proto_origin)

                if self.distance_mode == "cat":
                    e_types = torch.cat([e_out_mask, types_mask, (e_out - types_mask).abs()], -1)
                    e_types = self.dis_cls(e_types)
                    e_types = e_types[:, :, :0]
                elif self.distance_mode == "l2":
                    e_types = -(torch.pow(e_out_mask - types_mask, 2)).sum(-1)
                elif self.distance_mode == "cos":
                    sim_k = sim_k if sim_k else self.similar_k
                    e_types = sim_k * (e_out_mask * types_mask).sum(-1) / 768

                e_logits_mask = e_types
                if M:
                    em = e_type_mask.clone()
                    em[em.sum(-1) == 0] = 1
                    e = e_types * em
                    e_type_label = torch.zeros((B, M)).to(e_types.device)
                    type_loss_mask = self.calc_loss(
                        self.type_loss, e, e_type_label, e_type_mask[:, :, 0])
                    type_loss=type_loss+0.2*type_loss_mask #直接通过mask原型计算loss，不加可以注释掉
                    if self.proto_cL_Loss:   
                        type_loss = 0.2 * proto_cl_loss + type_loss #mask原型的cl loss

            # else:
            #     type_loss = torch.tensor(0).to(sequence_output.device) #这句先注释掉，目前不走
        else:
            e_logits, type_loss = None, None

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
    
    def get_mask_ids(self, input_ids, e_mask, e_type_mask):
        B, M, T = e_mask.shape
        input_ids_clone=input_ids.clone()
        input_ids_clone.unsqueeze(1).expand(B,M,T)
        e_mask_clone=e_mask.clone()
        # e_mask2=torch.zeros([T])
        e_mask_all=[]
        for i in range(0,B):  #构造一个全实体位置的矩阵
            for j in range(0,M):
                if j ==0:
                    e_mask2=e_mask_clone[i][j]
                if j>0:
                    e_mask2=e_mask2+e_mask_clone[i][j]
            e_mask_all.append(e_mask2.clone())
        e_mask_all=torch.stack(e_mask_all)  
        e_mask_all=e_mask_all.unsqueeze(1).expand(B,M,T)   #构造完成
        e_type_mask_clone=e_type_mask.clone()    #e_mask_all要与e_mask_clone对齐
        e_type_mask_clone=e_type_mask_clone[:,:,0:1].repeat(1, 1, T)
        e_mask_other = e_mask_all*e_type_mask_clone - e_mask_clone  #除了目标实体，mask了其他实体

        mask_ids = input_ids.clone().unsqueeze(1).expand(B,M,T) 
        '''构造e_mask_other的反转矩阵'''
        e_mask_other_convert = e_mask_other.clone()
        e_mask_other_convert[e_mask_other_convert==1]=2
        e_mask_other_convert[e_mask_other_convert==0]=1
        e_mask_other_convert[e_mask_other_convert==2]=0

        mask_ids = mask_ids * e_mask_other_convert + e_mask_other * 103
        return mask_ids

    def get_enity_hidden( #最终得到batch size * max entity * hidden_size的句向量
        self, hidden: torch.Tensor, e_mask: torch.Tensor, entity_mode: str
    ):      
        # import pdb
        # pdb.set_trace()
        B, M, T = e_mask.shape
        #hidden.unsqueeze(1).expand(B, M, T, -1)即在sequence前面加了一层max emtity，乘后项e_mask（只有实体词置1，其余为0的矩阵），最后得到
        #batch_size x max_entity_num x seq_len x hidden_size，其中seq_len只有实体词有向量，非实体词都为0.
        e_out = hidden.unsqueeze(1).expand(B, M, T, -1) * e_mask.unsqueeze(    
            -1
        )  # batch_size x max_entity_num x seq_len x hidden_size
        

        if entity_mode == "mean": #对seq_len那一维度加和，相当于现在e_out为每个句子实体词的表征
            return e_out.sum(2) / (e_mask.sum(-1).unsqueeze(-1) + 1e-30)  # batch_size x max_entity_num x hidden_size
    
    def get_enity_hidden_mask(
        self, hidden: torch.Tensor, e_mask: torch.Tensor, entity_mode: str
    ):      
        # import pdb
        # pdb.set_trace()
        # B, M, T = e_mask.shape
        '''hidden是我们构造的矩阵'''
        e_out = hidden * e_mask.unsqueeze(-1)  # batch_size x max_entity_num x seq_len x hidden_size

        

        if entity_mode == "mean":
            return e_out.sum(2) / (e_mask.sum(-1).unsqueeze(-1) + 1e-30)  # batch_size x max_entity_num x hidden_size
    
    def set_hidden(self,e_out, e_type_ids, e_type_mask,entity_types):  #对比学习时数据预处理，去除重复类别的数据，返回去重的embedding，以及对应的原型
        hiddens = e_out[e_type_mask[:, :, 0] == 1]
        labels = e_type_ids[:, :, 0][e_type_mask[:, :, 0] == 1]
        # import pdb
        # pdb.set_trace()
        set_labels = []  #去重的labels
        set_hiddens=[]   #去重的hiddens
        # hiddens,labels =list(hiddens),list(labels)
        # import pdb
        # pdb.set_trace()
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

    def set_proto(self, labels, entity_types): #得到对比学习所需要的原型,只取与之前方法对应的
        proto=[]
        for i in labels:
            proto.append(entity_types.types_embedding.weight.data[i])
            # import pdb
            # pdb.set_trace()
        return torch.stack(proto)
    def set_proto_mask(self, labels, entity_types): #得到对比学习所需要的原型,只取与之前方法对应的
        proto=[]
        for i in labels:
            proto.append(entity_types.types_mask_embedding.weight.data[i])
            # import pdb
            # pdb.set_trace()
        return torch.stack(proto)
    def set_proto_context(self, labels, entity_types): #得到对比学习所需要的原型,只取与之前方法对应的
        proto=[]
        for i in labels:
            proto.append(entity_types.types_context_embedding.weight.data[i])
            # import pdb
            # pdb.set_trace()
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
        # import pdb
        # pdb.set_trace()
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
