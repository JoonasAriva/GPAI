import torch
import torch.nn.functional as F

from modules.ScoreCAM.cam.basecam import BaseCAM


class ScoreCAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, retain_graph=False):
        _, b, c, h, w = input.size()

        # predication on raw input
        Y_prob, Y_hat, att = self.model_arch(input)  # .cuda()

        # if class_idx is None:
        #     predicted_class = logit.max(1)[-1]
        #     score = logit[:, logit.max(1)[-1]].squeeze()
        # else:
        #     predicted_class = torch.LongTensor([class_idx])
        #     score = logit[:, class_idx].squeeze()
        #
        # logit = F.softmax(logit)

        if torch.cuda.is_available():
            # predicted_class= predicted_class.cuda()
            att = att.cuda()
            Y_hat = Y_hat.cuda()
            print("y prob shape: ", Y_prob.shape)
            Y_prob = Y_prob.cuda()
            # logit = logit.cuda()
        # print(score)
        self.model_arch.zero_grad()
        # score.backward(retain_graph=retain_graph)
        Y_prob.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()
        print("activations shape: ", activations.shape)
        score_saliency_map = torch.zeros((b, 1, h, w))
        print("first sal map print shape: ", score_saliency_map.shape)
        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                new_Y_prob, _, new_attention = self.model_arch(input * norm_saliency_map)
                # output = F.softmax(output)
                score = new_Y_prob
                # print("debug (score, sal map): ", score.shape, saliency_map.shape)
                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            print("min = max")
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map, Y_hat, att

    def __call__(self, input, retain_graph=False):
        return self.forward(input, retain_graph)


class ScoreCAM_for_attention(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict, wrt_classifier_score=False):
        super().__init__(model_dict)
        self.classifier_score = wrt_classifier_score

    def forward(self, input, retain_graph=False):

        b, c, h, w = input.size()


        self.model_arch.zero_grad()
        Y_prob, Y_hat, unnorm_attention, cls_scores = self.model_arch(input, return_unnorm_attention=True,
                                                                      full_pass=True)


        if torch.isnan(unnorm_attention).any():
            print("unorm attention contains nans")

        top_att = unnorm_attention.cpu().flatten().argsort()
        important_slice_indices = top_att[-6:]  # we visualize 6 biggest
        print("important slices:", important_slice_indices.shape)



        activations = self.activations['value']
        b, k, u, v = activations.size()
        activations = activations[important_slice_indices]

        print("activations shape: ", activations.shape)
        print("k value:", k)

        score_saliency_map = torch.zeros((6, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()
        print("all acts: ", activations.shape)
        print("debug input shape: ", input.shape)
        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)

                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                # print("sal map, activcations in loop: ", saliency_map.shape)
                # normalize to 0-1
                scorecam_base = torch.zeros_like(torch.squeeze(saliency_map))

                min_per_channel = saliency_map.min(dim=2).values.min(dim=2).values.view(-1, 1, 1, 1)
                max_per_channel = saliency_map.max(dim=2).values.max(dim=2).values.view(-1, 1, 1, 1)
                # good_activations = min_per_channel != max_per_channel
                # good_activations = good_activations.view(-1)

                norm_saliency_map = (saliency_map - min_per_channel) / (max_per_channel - min_per_channel)

                # how much increase if keeping the highlighted region
                # predication on masked input
                if torch.isnan(norm_saliency_map).any():
                    print("norm saliency map contains nans: ")
                filtered_input = input[important_slice_indices] * norm_saliency_map

                _, _, new_attention = self.model_arch(filtered_input, return_unnorm_attention=True,
                                                      scorecam_wrt_classifier_score=self.classifier_score)

                # output = F.softmax(output)
                new_attention = torch.squeeze(new_attention).view(-1, 1, 1)
                if torch.isnan(filtered_input).any():
                    print("filtered input contains nans: ")

                if torch.isnan(new_attention).any():
                    print("new attention contains nans: ")
                    print(new_attention)

                scorecam_base += new_attention * torch.squeeze(saliency_map)
                score_saliency_map += scorecam_base

        #print("nan after loop?: ", torch.isnan(score_saliency_map).any())


        score_saliency_map = F.relu(score_saliency_map)

        min_per_channel = score_saliency_map.min(dim=1).values.min(dim=1).values.view(-1, 1, 1)

        max_per_channel = score_saliency_map.max(dim=1).values.max(dim=1).values.view(-1, 1, 1)


        # good_activations = good_activations.view(-1)

        score_saliency_map = (score_saliency_map - min_per_channel) / (max_per_channel - min_per_channel)
        print("nan after final normalization?: ", torch.isnan(score_saliency_map).any())
        # print("nr of good activations: ", len(good_activations))
        # all_slices_saliency_map[idx, :, :] = score_saliency_map

        _, _, individual_predictions = self.model_arch(input[important_slice_indices],
                                                       scorecam_wrt_classifier_score=True)


        return score_saliency_map, Y_hat, unnorm_attention, cls_scores

    def __call__(self, input, retain_graph=False):
        return self.forward(input, retain_graph)
