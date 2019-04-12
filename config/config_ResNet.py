import os

GAMMA = 0.9
START_EPSILON = 0.2
END_EPSILON = 0.01
REPLAY_SIZE = 50000
BATCH_SIZE = 16
HIDDEN_SIZE = 20
MAX_EPISODES = 5000
MAX_STEPS = 300

out_dir = os.path.join('./', "runs", 'ResNet')
summary_dir = os.path.join(out_dir, "summaries")
checkpoint_dir = os.path.join(out_dir, "checkpoints")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


