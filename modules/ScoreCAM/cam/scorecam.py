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

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, scan_end, depth_scores=None, retain_graph=False, tumor=True):

        b, c, h, w = input.size()

        self.model_arch.zero_grad()
        out = self.model_arch(input, scan_end=scan_end, cam=True)  # depth_scores=depth_scores)

        scores = out["scores"]

        # scores = out["attention_weights"]
        # rel_scores = out["relevancy_scores"]
        important_slice_indices = scores.cpu().flatten().argsort()[:4]

        # position_scores, tumor_score, rel_scores

        # rel_labels = torch.where(
        #     (self.model_arch.depth_range[0].item() < position_scores) & (
        #             position_scores < self.model_arch.depth_range[1].item()),
        #     1, 0)

        # rel_labels = rel_labels.cpu().flatten()  # .argsort()

        # imp = torch.where(rel_labels == 1)[0]
        # print("imp shape: ", imp.shape)
        #
        # important_slice_indices = imp

        activations = self.activations['value']
        b, k, u, v = activations.size()
        activations = activations[important_slice_indices]  # important_slice_indices]

        # print("len:", len(important_slice_indices))
        score_saliency_map = torch.zeros((len(important_slice_indices), h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()
        maxest_item = -100
        minest_item = 100
        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)

                if saliency_map.max() == saliency_map.min():
                    print(i, "min = max, skipping")
                    min_per_channel = saliency_map.min(dim=2).values.min(dim=2).values.view(-1, 1, 1, 1)
                    max_per_channel = saliency_map.max(dim=2).values.max(dim=2).values.view(-1, 1, 1, 1)
                    # print("min and max per channel", min_per_channel, max_per_channel)
                    continue
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                if torch.isnan(saliency_map).any():
                    print("interpolated saliency map contains nans")
                # print("sal map, activations in loop: ", saliency_map.shape)
                # normalize to 0-1
                score_saliency_map = torch.zeros_like(torch.squeeze(saliency_map))

                min_per_channel = saliency_map.min(dim=2).values.min(dim=2).values.view(-1, 1, 1, 1)
                max_per_channel = saliency_map.max(dim=2).values.max(dim=2).values.view(-1, 1, 1, 1)

                norm_saliency_map = (saliency_map - min_per_channel) / (max_per_channel - min_per_channel)

                # how much increase if keeping the highlighted region
                # predication on masked input
                if torch.isnan(norm_saliency_map).any():
                    print("norm saliency map contains nans: ")

                filtered_input = input[important_slice_indices] * norm_saliency_map

                # THIS PART IS FOR SIMPLE MIL MODEL
                out = self.model_arch(filtered_input, scan_end=len(important_slice_indices), cam=True)
                score = out["attention_weights"]
                #score = out["scores"]

                new_attention = torch.squeeze(score).view(-1, 1, 1)

                if torch.isnan(filtered_input).any():
                    print("filtered input contains nans: ")

                if torch.isnan(new_attention).any():
                    print("new attention contains nans: ")
                # print(new_attention)
                maxitem = torch.max(new_attention).item()
                minitem = torch.min(new_attention).item()

                if maxitem > maxest_item:
                    maxest_item = maxitem
                if minitem < minest_item:
                    minest_item = minitem
                score_saliency_map += new_attention * torch.squeeze(saliency_map)

                if torch.isnan(score_saliency_map).any():
                    print("score sal map contains nans: ")

                # print("nan after loop?: ", torch.isnan(score_saliency_map).any())
                # print("looking for tumor?: ", tumor)
                # print("max and min:", maxest_item, minest_item)
                # if tumor:
                #    score_saliency_map = F.relu(score_saliency_map)

                min_per_channel = score_saliency_map.min(dim=1).values.min(dim=1).values.view(-1, 1, 1)

                max_per_channel = score_saliency_map.max(dim=1).values.max(dim=1).values.view(-1, 1, 1)
                # print("mins and maxs per channel (6):", min_per_channel, max_per_channel)
                # good_activations = good_activations.view(-1)

                if torch.isnan(score_saliency_map).any():
                    print("score sal map contains nans2: ")

                score_saliency_map = (score_saliency_map - min_per_channel) / (max_per_channel - min_per_channel)

                if torch.isnan(score_saliency_map).any():
                    print("score sal map contains nans3: ")

        return score_saliency_map, important_slice_indices, scores,  # rel_scores[important_slice_indices]

    def __call__(self, input, scan_end, retain_graph=False, tumor=True):
        return self.forward(input, scan_end, retain_graph, tumor)
