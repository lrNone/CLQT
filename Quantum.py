import torch
from torch import nn
import torch.nn.functional as F



class DynamicEntanglement(nn.Module):
    def __init__(self, embed_dim, num_languages=5, device=torch.device('cuda')):
        super(DynamicEntanglement, self).__init__()
        self.embed_dim = embed_dim
        self.num_languages = num_languages
        self.device = device

        # 模拟 Hadamard 门：线性变换生成叠加态
        self.hadamard = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.hadamard.weight)

        # 模拟 CRY 门：参数化旋转
        self.cry_angles = nn.Parameter(torch.zeros(num_languages * (num_languages - 1) // 2))

    def forward(self, lang_reps, weights):
        """
        输入：
            lang_reps: (bsz, num_languages, seq_len, embed_dim)，每种语言的表示
            weights: (num_languages, num_languages)，语言间相似性权重
        输出：
            entangled_reps: (bsz, seq_len, embed_dim)，纠缠后的共享表示
        """
        bsz, num_langs, seq_len, emb_dim = lang_reps.shape
        assert num_langs == self.num_languages

        # 模拟 Hadamard 门：生成叠加态
        lang_reps = self.hadamard(lang_reps)  # (bsz, num_langs, seq_len, embed_dim)

        # 计算 CRY 门的旋转角度（基于 weights）
        cry_idx = 0
        entangled_states = []
        for i in range(num_langs):
            state_i = lang_reps[:, i]  # (bsz, seq_len, embed_dim)
            for j in range(i + 1, num_langs):
                w_ij = weights[i, j]
                phi_ij = torch.arcsin(w_ij) + self.cry_angles[cry_idx]  # 可训练偏置
                cry_idx += 1

                # 模拟 CRY 门：旋转 state_j
                state_j = lang_reps[:, j]
                cos_phi = torch.cos(phi_ij / 2)
                sin_phi = torch.sin(phi_ij / 2)
                rotated_j = cos_phi * state_j - sin_phi * state_i  # 简化的旋转操作

                # 加权组合
                entangled_states.append(w_ij * (state_i + rotated_j))

        # 归一化纠缠态
        entangled_reps = torch.stack(entangled_states, dim=0).sum(dim=0)  # (bsz, seq_len, embed_dim)
        entangled_reps = F.normalize(entangled_reps, dim=-1)

        return entangled_reps

class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim=1, zero_phase=False, device=torch.device('cuda')):
        super(PositionEmbedding, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.zero_phase = zero_phase


        frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim), embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix)

        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix)

        # self.frequencies = nn.Parameter(frequency_inits.unsqueeze(dim = 0).to(self.device))

    def forward(self, x):

        # No speaker embedding
        if self.input_dim == 1:
            x = torch.zeros_like(x)
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)

        time_stamps = x.shape[1]

        positions = torch.arange(time_stamps).unsqueeze(-1).to(self.device)
        pos_embed = positions.repeat(1, self.embed_dim) * self.frequency_embedding(x) + phases
        if self.zero_phase:
            pos_embed = torch.zeros_like(pos_embed)
        # batch_pos_embed = pos_embed.unsqueeze(dim = 0).expand_as(x)

        return pos_embed


class ComplexMultiply(torch.nn.Module):
    def __init__(self):
        super(ComplexMultiply, self).__init__()

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        phase = inputs[0]
        amplitude = inputs[1]

        if amplitude.dim() == phase.dim() + 1:  # Assigning each dimension with same phase
            cos = torch.unsqueeze(torch.cos(phase), dim=-1)
            sin = torch.unsqueeze(torch.sin(phase), dim=-1)

        elif amplitude.dim() == phase.dim():  # Each dimension has different phases
            cos = torch.cos(phase)
            sin = torch.sin(phase)


        else:
            raise ValueError('input dimensions of phase and amplitude do not agree to each other.')

        real_part = cos * amplitude
        imag_part = sin * amplitude

        return [real_part, imag_part]


class QOuter(torch.nn.Module):
    def __init__(self):
        super(QOuter, self).__init__()

    def forward(self, x):

        if not isinstance(x, list):
            raise ValueError('xr should be called '
                             'on a list of 2 inputs.')

        if len(x) != 2:
            raise ValueError('x should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(x)) + ' inputs.')

        # x[0], x[1] has shape:
        # (batch_size, time_stamps, embedding_dim)
        real = x[0].transpose(0, 1)
        imag = x[1].transpose(0, 1)
        output = []
        for r, i in zip(real, imag):
            output_rr = []
            output_ii = []
            for rr, ii in zip(r, i):
                unsqueezed_rr = torch.unsqueeze(rr, dim=-1)
                unsqueezed_ii = torch.unsqueeze(ii, dim=-1)
                _r = torch.mm(unsqueezed_rr, unsqueezed_rr.t()) + torch.mm(unsqueezed_ii, unsqueezed_ii.t())
                _i = -torch.mm(unsqueezed_rr, unsqueezed_ii.t()) + torch.mm(unsqueezed_ii, unsqueezed_rr.t())

                output_rr.append(_r)
                output_ii.append(_i)

            output_rr = torch.stack(output_rr, dim=0)
            output_ii = torch.stack(output_ii, dim=0)
            output.append([output_rr, output_ii])

        return output


class QMeasurement(torch.nn.Module):
    def __init__(self, embed_dim, device=torch.device('cuda')):
        super(QMeasurement, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.kernel = torch.nn.Parameter(
            torch.stack([torch.eye(embed_dim).to(self.device), torch.zeros(embed_dim, embed_dim).to(self.device)],
                        dim=-1))

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        input_real = inputs[0]
        input_imag = inputs[1]

        real_kernel = self.kernel[:, :, 0]
        imag_kernel = self.kernel[:, :, 1]

        real_kernel = real_kernel.unsqueeze(-1)
        imag_kernel = imag_kernel.unsqueeze(-1)

        projector_real = torch.matmul(real_kernel, real_kernel.transpose(1, 2)) \
                         + torch.matmul(imag_kernel, imag_kernel.transpose(1, 2))
        projector_imag = torch.matmul(imag_kernel, real_kernel.transpose(1, 2)) \
                         - torch.matmul(real_kernel, imag_kernel.transpose(1, 2))
        # only real part is non-zero
        # input_real.shape = [batch_size, seq_len, embed_dim, embed_dim] or [batch_size, embed_dim, embed_dim]
        # projector_real.shape = [num_measurements, embed_dim, embed_dim]
        output_real = torch.matmul(torch.flatten(input_real, start_dim=-2, end_dim=-1),
                                   torch.flatten(projector_real, start_dim=-2, end_dim=-1).t()) \
                      - torch.matmul(torch.flatten(input_imag, start_dim=-2, end_dim=-1),
                                     torch.flatten(projector_imag, start_dim=-2, end_dim=-1).t())

        return output_real



class QMixture(torch.nn.Module):

    def __init__(self, use_weights=True, device=torch.device('cuda')):
        super(QMixture, self).__init__()
        self.use_weights = use_weights
        self.device = device

    def forward(self, inputs):

        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        in_modalities = inputs[0]  # [modal_1,...modal_n], each being a list of [real, imag] arrays

        weights = inputs[1].transpose(0, 1)  # (time_stamps, batch_size, num_modalities)
        embed_dim = in_modalities[0][0][0].shape[-1]
        outputs = []
        for reps_t in zip(*in_modalities, weights):
            multimodal_rep = [torch.stack(rep_field, dim=-1) for rep_field in zip(*reps_t[:-1])]
            w = reps_t[-1].unsqueeze(dim=1).unsqueeze(dim=-1).expand(-1, embed_dim, -1, -1)
            output_rep = [torch.matmul(_rep, w).squeeze(dim=-1) for _rep in multimodal_rep]
            outputs.append(output_rep)

        return outputs


class PositionEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, input_dim=1, zero_phase=False, device=torch.device('cuda')):
        super(PositionEmbedding, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.zero_phase = zero_phase

        # Vaswani et al.
        frequency_inits = 1 / torch.pow(10000, torch.true_divide(torch.arange(embed_dim), embed_dim))
        frequency_matrix = frequency_inits.repeat(self.input_dim, 1)
        self.frequency_embedding = nn.Embedding.from_pretrained(frequency_matrix)

        self.frequency_embedding.weight.requires_grad = True
        phase_matrix = torch.rand(self.input_dim, self.embed_dim)
        self.phase_embedding = nn.Embedding.from_pretrained(phase_matrix)
        self.phase_embedding.weight.requires_grad = True
        # self.frequencies = nn.Parameter(frequency_inits.unsqueeze(dim = 0).to(self.device))

    def forward(self, x):

        # No speaker embedding
        if self.input_dim == 1:
            x = torch.zeros_like(x)
        phases = self.phase_embedding(x)
        phases = 2 * 3.14 * nn.Sigmoid()(phases)

        time_stamps = x.shape[1]

        positions = torch.arange(time_stamps).unsqueeze(-1).to(self.device)
        pos_embed = positions.repeat(1, self.embed_dim) * self.frequency_embedding(x) + phases
        if self.zero_phase:
            pos_embed = torch.zeros_like(pos_embed)
        # batch_pos_embed = pos_embed.unsqueeze(dim = 0).expand_as(x)

        return pos_embed
