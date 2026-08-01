import sys
log = open('experiments/02_ppo_training/logs/train_bc_full_20260731_210235.log', 'w')
for line in iter(sys.stdin.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
    log.write(line)
    log.flush()
