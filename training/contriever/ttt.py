import torch

labels = [[0, 1], [1,2], [3], [0,3]]
scores = torch.tensor([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20]])
len_labels = [len(l) for l in labels]
labels = torch.tensor([l for ls in labels for l in ls], dtype=torch.long)
assert labels.dim() == 1
len_labels = torch.tensor(len_labels, dtype=torch.long)  
target_mask = torch.zeros(scores.size())

repeated = torch.repeat_interleave(scores, len_labels, dim=0, output_size=len_labels.sum())
print('repeated', repeated)

pos_mask = torch.zeros((repeated.size()))
pos_mask.scatter_(dim=1, index=labels.unsqueeze(1), src=torch.ones(labels.unsqueeze(1).size()))
print('pos mask', pos_mask)


start = 0
for i, _len in enumerate(len_labels):  # i -> batch index
    pos_indices = labels[start:start+_len]
    target_mask[i, pos_indices] = 1   # Assign positive labels
    start += _len
    


print('target mask', target_mask)
repeated_neg_mask = ~torch.repeat_interleave(target_mask.bool(), len_labels, dim=0, output_size=len_labels.sum())
print('repeated neg mask', repeated_neg_mask)
repeated_neg = repeated * repeated_neg_mask

print("repeated_neg", repeated_neg)

repeated_pos = repeated * pos_mask

print("repeated_pos", repeated_pos)
_sum = repeated_neg + repeated_pos
_sum[_sum == 0] = -float('inf')

print('sum', _sum)
loss = torch.nn.functional.cross_entropy(_sum, labels)
print('loss', loss)

losses = []
losses.append(torch.nn.functional.cross_entropy(torch.tensor([1,3,4,5]).unsqueeze(0).float(), torch.tensor([0])))
losses.append(torch.nn.functional.cross_entropy(torch.tensor([2,3,4,5]).unsqueeze(0).float(), torch.tensor([0])))
losses.append(torch.nn.functional.cross_entropy(torch.tensor([6,7,9,10]).unsqueeze(0).float(), torch.tensor([1])))
losses.append(torch.nn.functional.cross_entropy(torch.tensor([6,8,9,10]).unsqueeze(0).float(), torch.tensor([1])))
losses.append(torch.nn.functional.cross_entropy(torch.tensor([11,12,13,14,15]).unsqueeze(0).float(), torch.tensor([3])))
losses.append(torch.nn.functional.cross_entropy(torch.tensor([16,17,18,20]).unsqueeze(0).float(), torch.tensor([0])))
losses.append(torch.nn.functional.cross_entropy(torch.tensor([19,17,18,20]).unsqueeze(0).float(), torch.tensor([0])))
print(sum(losses)/len(losses))


