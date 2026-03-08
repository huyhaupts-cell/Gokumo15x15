import numpy as np
from GameEnv import GomokuEnv


class VectorizedGomokuEnv:
    
    def __init__(self, num_envs=32, board_size=15, win_condition=5):
        
        self.num_envs = num_envs
        
        self.envs = [
            GomokuEnv(board_size, win_condition)
            for _ in range(num_envs)
        ]
        
        self.board_size = board_size
        self.action_size = board_size * board_size

    def reset(self):
        
        obs = []
        infos = []
        
        for env in self.envs:
            o, info = env.reset()
            obs.append(o)
            infos.append(info)
            
        return np.stack(obs), infos

    def step(self, actions):
        
        obs = []
        rewards = []
        dones = []
        truncs = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            
            o, r, done, trunc, info = env.step(action)
            
            if done or trunc:
                info["final_observation"] = o
                final_info = info.copy()
                info["final_info"] = final_info
                o, reset_info = env.reset()
                
                info["action_mask"] = reset_info["action_mask"]
            
            obs.append(o)
            rewards.append(r)
            dones.append(done)
            truncs.append(trunc)
            infos.append(info)
        
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            np.array(truncs, dtype=np.bool_),
            infos
        )