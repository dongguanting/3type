import torch
import pdb


def cos_similarity(mat1, mat2, temperature):
    norm1 = torch.norm(mat1, p=2, dim=1).view(-1, 1)
    norm2 = torch.norm(mat2, p=2, dim=1).view(1, -1)
    cos_sim = torch.matmul(mat1, mat2.t()) / torch.matmul(norm1, norm2)
    norm_sim = torch.exp(cos_sim / temperature)
    return norm_sim


def info_NCE(sim_pos, sim_total):
    deno = torch.sum(sim_total) - sim_total[0][0] + sim_pos[0][0]
    loss = -torch.log(sim_pos[0][0] / deno)
    return loss


def contrastive_loss(plm_encoder, inputs_sentences, positive_examples):
    # 所有词的平均作为句子表征
    inputs = plm_encoder.encoder(inputs_sentences).last_hidden_state
    positive_examples = plm_encoder.encoder(positive_examples).last_hidden_state

    inputs_mean = torch.mean(inputs, dim=1)
    positive_examples_mean = torch.mean(positive_examples, dim=1)

    batch_size = inputs_mean.size()[0]
    cons_loss = 0
    for input, pos_exp in zip(inputs_mean, positive_examples_mean):
        input = torch.reshape(input, (1, -1))
        pos_exp = torch.reshape(pos_exp, (1, -1))
        sim_pos = cos_similarity(input, pos_exp, 0.1)
        sim_total = cos_similarity(input, inputs_mean, 0.1)
        cur_loss = info_NCE(sim_pos, sim_total)
        cons_loss += cur_loss

    return cons_loss / batch_size

# def CL_Loss(ori_hidden, aug_hidden):
#     inputs_mean = torch.mean(ori_hidden, dim=1)
#     positive_examples_mean = torch.mean(aug_hidden, dim=1)

#     batch_size = inputs_mean.size()[0]
#     cons_loss = 0
#     for ori_input, pos_exp in zip(inputs_mean, positive_examples_mean):
#         ori_input = torch.reshape(ori_input, (1, -1))
#         pos_exp = torch.reshape(pos_exp, (1, -1))
#         sim_pos = cos_similarity(ori_input, pos_exp, 0.1)
#         sim_total = cos_similarity(ori_input, inputs_mean, 0.1)
#         cur_loss = info_NCE(sim_pos, sim_total)
#         cons_loss += cur_loss

#     return cons_loss / batch_size

def instance_CL_Loss(ori_hidden, aug_hidden, type="origin", temp=0.5):
    inputs_mean = ori_hidden
    positive_examples_mean = aug_hidden
    batch_size = inputs_mean.size()[0]
    cons_loss = 0
    count = 0
    for ori_input, pos_exp in zip(inputs_mean, positive_examples_mean):
        ori_input = torch.reshape(ori_input, (1, -1))
        pos_exp = torch.reshape(pos_exp, (1, -1))
        sim_pos = cos_similarity(ori_input, pos_exp, temp)
        '''魔改对比学习，让他去和其余的负样本对比'''
        # negative=torch.Tensor()
        if type == "origin":
            negative = inputs_mean[torch.arange(positive_examples_mean.size(0)) != count]  # batch中其他样本作为negative
        elif type == "aug":
            negative = positive_examples_mean[torch.arange(positive_examples_mean.size(0)) != count]  # 和其他batch样本的keyword 拉大
        count += 1

        sim_total = cos_similarity(ori_input, negative, temp)  ##修改为了正样例的平均
        cur_loss = info_NCE(sim_pos, sim_total)
        cons_loss += cur_loss

    return cons_loss / batch_size
def proto_CL_Loss(ori_hidden, aug_hidden, type="origin", temp=0.5):
    inputs_mean = ori_hidden
    positive_examples_mean = aug_hidden
    batch_size = inputs_mean.size()[0]
    cons_loss = 0
    count = 0
    for ori_input, pos_exp in zip(inputs_mean, positive_examples_mean):
        ori_input = torch.reshape(ori_input, (1, -1))
        pos_exp = torch.reshape(pos_exp, (1, -1))
        sim_pos = cos_similarity(ori_input, pos_exp, temp)
        '''魔改对比学习，让他去和其余的负样本对比'''
        # negative=torch.Tensor()
        if type == "origin":
            negative = inputs_mean[torch.arange(positive_examples_mean.size(0)) != count]  # batch中其他样本作为negative
        elif type == "aug":
            negative = positive_examples_mean[torch.arange(positive_examples_mean.size(0)) != count]  # 和其他batch样本的keyword 拉大
        count += 1

        sim_total = cos_similarity(ori_input, negative, temp)  ##修改为了正样例的平均
        cur_loss = info_NCE(sim_pos, sim_total)
        cons_loss += cur_loss

    return cons_loss / batch_size



    

# if __name__ == "__main__":
#     a = torch.rand()
