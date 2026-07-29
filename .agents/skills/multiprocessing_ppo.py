import multiprocessing as mp
import time

def rollout_worker(worker_id, env_maker, policy_net, weights_queue, trajectory_queue, stop_event, rollout_steps):
    env = env_maker()
    
    while not stop_event.is_set():
        # Update weights if new ones are broadcasted
        try:
            while not weights_queue.empty():
                new_weights = weights_queue.get_nowait()
                policy_net.set_weights(new_weights)
        except Exception:
            pass

        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        state = env.reset()
        
        for _ in range(rollout_steps):
            action, log_prob, value = policy_net.get_action_and_value(state)
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state
            if done:
                state = env.reset()
                
        # Send gathered rollout data to the main process
        trajectory_queue.put({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': log_probs,
            'dones': dones
        })

class DistributedPPO:
    def __init__(self, num_workers, env_maker, policy_net_class, rollout_steps):
        self.num_workers = num_workers
        self.env_maker = env_maker
        self.policy_net_class = policy_net_class
        self.rollout_steps = rollout_steps
        
        # Dedicated queue per worker to broadcast weights without race conditions
        self.weights_queues = [mp.Queue() for _ in range(num_workers)]
        self.trajectory_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.workers = []
        
    def start(self):
        for i in range(self.num_workers):
            policy_net = self.policy_net_class()
            p = mp.Process(
                target=rollout_worker,
                args=(i, self.env_maker, policy_net, self.weights_queues[i], 
                      self.trajectory_queue, self.stop_event, self.rollout_steps)
            )
            p.start()
            self.workers.append(p)
            
    def broadcast_weights(self, weights):
        for q in self.weights_queues:
            q.put(weights)
            
    def collect_trajectories(self, expected_trajectories):
        trajectories = []
        for _ in range(expected_trajectories):
            trajectories.append(self.trajectory_queue.get())
        return trajectories
        
    def stop(self):
        self.stop_event.set()
        for w in self.workers:
            w.join()
