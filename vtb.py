def get_iu(box, grt):
    # box, grt: [x1, y1, x2, y2]
    box, grt = list(map(float, box)), list(map(float, grt))
    inter_x = max(min(box[2], grt[2])-max(box[0], grt[0])+1, 0)
    inter_y = max(min(box[3], grt[3])-max(box[1], grt[1])+1, 0)
    inter_area = inter_x * inter_y
    union_area = (box[2]-box[0]+1)*(box[3]-box[1]+1)+(grt[2]-grt[0]+1)*(grt[3]-grt[1]+1)-inter_area
    iu = inter_area/union_area
    return iu


def vtb(boxes, scores, lambda_=1):
    T = len(boxes)
    V, I = [[] for _ in range(T)], [[] for _ in range(T)]
    # Initialization
    for i in range(len(boxes[0])):
        V[0].append(scores[0][i])
        I[0].append(i)
    # Iter
    for t in range(1, T):
        for i in range(len(boxes[t])):
            box_i, score_i = boxes[t][i], scores[t][i]
            vti = []
            for j in range(len(boxes[t-1])):
                box_j = boxes[t-1][j]
                iu = get_iu(box_j, box_i)
                s_iu = iu if iu > 0.1 else -1000
                vti_j = V[t-1][j]+lambda_*s_iu
                vti.append(vti_j)
            max_i, max_val = vti.index(max(vti)), max(vti)
            V[t].append(max_val+score_i)
            I[t].append(max_i)
    max_idx = [-1 for _ in range(T)]
    max_idx[-1] = V[-1].index(max(V[-1]))
    for t in range(T-1, 0, -1):
        last_idx = max_idx[t]
        max_idx[t-1] = I[t][last_idx]
    box_coords = []
    for t in range(T):
        box_coords.append([boxes[t][max_idx[t]][0], boxes[t][max_idx[t]][1], boxes[t][max_idx[t]][2], boxes[t][max_idx[t]][3]])
    return box_coords
