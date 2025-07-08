import torch

def generate_sp_matrices(joint_hierarchy):
    # 获取所有关节列表
    joints = list(joint_hierarchy.keys())
    n_joints = len(joints)

    # 初始化最短路径矩阵
    sp_matrix = [[0]*n_joints for _ in range(n_joints)]

    # 预计算每个节点到根节点的路径
    paths_to_root = {}
    for joint in joints:
        path = []
        current = joint
        while current is not None:
            path.append(current)
            current = joint_hierarchy[current]
        paths_to_root[joint] = path

    # 填充矩阵
    for i in range(n_joints):
        for j in range(n_joints):
            if i == j:
                continue

            # 获取两个关节的路径
            path_i = paths_to_root[joints[i]]
            path_j = paths_to_root[joints[j]]

            # 找到最近的公共祖先
            lca = None
            for node in path_i:
                if node in path_j:
                    lca = node
                    break

            # 计算距离 = i到LCA的步数 + j到LCA的步数
            distance = path_i.index(lca) + path_j.index(lca)
            sp_matrix[i][j] = distance

    return torch.Tensor(sp_matrix)

if __name__ == "__main__":
    joint_hierarchy = {
        'root': None,    # 躯干作为根节点
        'FL_hip': 'root',    # 左前腿-髋关节
        'FL_thigh': 'FL_hip',  # 左前腿-大腿
        'FL_calf': 'FL_thigh', # 左前腿-小腿
        'FR_hip': 'root',     # 右前腿-髋关节
        'FR_thigh': 'FR_hip',  # 右前腿-大腿
        'FR_calf': 'FR_thigh', # 右前腿-小腿
        'RL_hip': 'root',     # 左后腿-髋关节
        'RL_thigh': 'RL_hip',  # 左后腿-大腿
        'RL_calf': 'RL_thigh', # 左后腿-小腿
        'RR_hip': 'root',     # 右后腿-髋关节
        'RR_thigh': 'RR_hip',  # 右后腿-大腿
        'RR_calf': 'RR_thigh'  # 右后腿-小腿
    }
    # 测试生成
    SP_MATRICES = generate_sp_matrices(joint_hierarchy)
    print(SP_MATRICES)


    H1_SHORTPATH_MATRICES = torch.Tensor(
            [[0, 1, 1, 1,     2, 2, 2, 2,      3, 3, 3, 3,     4, 4, 4, 4,      5, 5, 5, 5],    # 0  ROOT
            [1, 0, 2,  2,     1, 3, 3, 3,      2, 4, 4, 4,     3, 5, 5, 5,      4, 6, 6, 6],     # 1  left_hip_yaw_joint
            [1, 2, 0,  2,     3, 1, 3, 3,      4, 2, 4, 4,     5, 3, 5, 5,      6, 4, 6, 6],     # 2  right_hip_yaw_joint
            [1, 2, 2,  0,     3, 3, 1, 1,      4, 4, 2, 2,     5, 5, 3, 3,      6, 6, 4, 4],     # 3  torso_joint

            [2, 1, 3,  3,     0, 4, 4, 4,      1, 5, 5, 5,     2, 6, 6, 6,      3, 7, 7, 7],     # 4  left_hip_roll_joint
            [2, 3, 1,  3,     4, 0, 4, 4,      5, 1, 5, 5,     6, 2, 6, 6,      7, 3, 7, 7],     # 5  right_hip_roll_joint
            [2, 3, 3,  1,     4, 4, 0, 2,      5, 5, 1, 3,     6, 6, 2, 4,      7, 7, 3, 5],     # 6  left_shoulder_pitch_joint
            [2, 3, 3,  1,     4, 4, 2, 0,      5, 5, 3, 1,     6, 6, 4, 2,      7, 7, 5, 3],     # 7  right_shoulder_pitch_joint

            [3, 2, 4,  4,     1, 5, 5, 5,      0, 6, 6, 6,     1, 7, 7, 7,      2, 8, 8, 8],     # 8  left_hip_pitch_joint
            [3, 4, 2,  4,     5, 1, 5, 5,      6, 0, 6, 6,     7, 1, 7, 7,      8, 2, 8, 8],     # 9  right_hip_pitch_joint
            [3, 4, 4,  2,     5, 5, 1, 3,      6, 6, 0, 4,     7, 7, 1, 5,      8, 8, 2, 6],     # 10  left_shoulder_roll_joint
            [3, 4, 4,  2,     5, 5, 3, 1,      6, 6, 4, 0,     7, 7, 5, 1,      8, 8, 6, 2],     # 11  right_shoulder_roll_joint

            [4, 3, 5,  5,     2, 6, 6, 6,      1, 7, 7, 7,     0, 8, 8, 8,      1, 9, 9, 9],     # 12  left_knee_joint
            [4, 5, 3,  5,     6, 2, 6, 6,      7, 1, 7, 7,     8, 0, 8, 8,      9, 1, 9, 9],     # 13  right_knee_joint
            [4, 5, 5,  3,     6, 6, 2, 4,      7, 7, 1, 5,     8, 8, 0, 6,      9, 9, 1, 7],     # 14  left_shoulder_yaw_joint
            [4, 5, 5,  3,     6, 6, 4, 2,      7, 7, 5, 1,     8, 8, 6, 0,      9, 9, 7, 1],     # 15  right_shoulder_yaw_joint

            [5, 4, 6,  6,     3, 7, 7, 7,      2, 8, 8, 8,     1, 9, 9, 9,      0,10,10,10],     # 16  left_ankle_joint
            [5, 6, 4,  6,     7, 3, 7, 7,      8, 2, 8, 8,     9, 1, 9, 9,     10, 0,10,10],     # 17  right_ankle_joint
            [5, 6, 6,  4,     7, 7, 3, 5,      8, 8, 2, 6,     9, 9, 1, 7,     10,10, 0, 8],     # 18  left_elbow_joint
            [5, 6, 6,  4,     7, 7, 5, 3,      8, 8, 6, 2,     9, 9, 7, 1,     10,10, 8, 0]]     # 19  right_elbow_joint
    )

    H1_hierarchy = {
            'root': None,
            'left_hip_yaw_joint': 'root',
            'right_hip_yaw_joint': 'root',
            'torso_joint': 'root',

            'left_hip_roll_joint': 'left_hip_yaw_joint',
            'right_hip_roll_joint': 'right_hip_yaw_joint',
            'left_shoulder_pitch_joint': 'torso_joint',
            'right_shoulder_pitch_joint': 'torso_joint',

            'left_hip_pitch_joint': 'left_hip_roll_joint',
            'right_hip_pitch_joint': 'right_hip_roll_joint',
            'left_shoulder_roll_joint': 'left_shoulder_pitch_joint',
            'right_shoulder_roll_joint': 'right_shoulder_pitch_joint',

            'left_knee_joint': 'left_hip_pitch_joint',
            'right_knee_joint': 'right_hip_pitch_joint',
            'left_shoulder_yaw_joint': 'left_shoulder_roll_joint',
            'right_shoulder_yaw_joint': 'right_shoulder_roll_joint',

            'left_ankle_joint': 'left_knee_joint',
            'right_ankle_joint': 'right_knee_joint',
            'left_elbow_joint': 'left_shoulder_yaw_joint',
            'right_elbow_joint': 'right_shoulder_yaw_joint',
        }



    SP_MATRICES = generate_sp_matrices(H1_hierarchy)
    print(SP_MATRICES)

    print(SP_MATRICES - H1_SHORTPATH_MATRICES)
