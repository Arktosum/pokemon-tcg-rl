import streamlit as st
import pandas as pd
import json
import time
import os

st.set_page_config(page_title="PPO Training Dashboard", layout="wide")
st.title("PPO Training Dashboard")

log_path = 'logs/metrics.jsonl'

if not os.path.exists(log_path):
    st.warning("Logs not found yet. Waiting...")
    time.sleep(2)
    st.rerun()

gameplay_data = []
network_data = []

with open(log_path, 'r') as f:
    for line in f:
        try:
            d = json.loads(line.strip())
            if d.get("type") == "gameplay":
                gameplay_data.append(d)
            elif d.get("type") == "network":
                network_data.append(d)
        except:
            pass

df_gameplay = pd.DataFrame(gameplay_data)
df_network = pd.DataFrame(network_data)

if not df_gameplay.empty:
    df_gameplay['timestamp'] = pd.to_datetime(df_gameplay['timestamp'])
    df_gameplay = df_gameplay.sort_values('episode')
    
    # Calculate rolling 10-episode averages
    df_gameplay['rolling_reward'] = df_gameplay['reward'].rolling(10, min_periods=1).mean()
    df_gameplay['rolling_win_rate'] = df_gameplay['win'].rolling(10, min_periods=1).mean()
    df_gameplay['rolling_steps'] = df_gameplay['steps'].rolling(10, min_periods=1).mean()

    col1, col2, col3 = st.columns(3)
    latest_win_rate = df_gameplay['rolling_win_rate'].iloc[-1]
    latest_reward = df_gameplay['rolling_reward'].iloc[-1]
    latest_steps = df_gameplay['rolling_steps'].iloc[-1]
    
    col1.metric("Rolling Win Rate (10 ep)", f"{latest_win_rate:.2%}")
    col2.metric("Rolling Reward (10 ep)", f"{latest_reward:.2f}")
    col3.metric("Rolling Steps (10 ep)", f"{latest_steps:.1f}")
    
    st.subheader("Gameplay Metrics (Rolling Averages)")
    st.line_chart(df_gameplay.set_index('episode')[['rolling_reward', 'rolling_win_rate', 'rolling_steps']])

if not df_network.empty:
    df_network['timestamp'] = pd.to_datetime(df_network['timestamp'])
    df_network = df_network.sort_values('episode')
    
    st.subheader("Network Losses")
    st.line_chart(df_network.set_index('episode')[['actor_loss', 'critic_loss']])
    
    st.subheader("Entropy")
    st.line_chart(df_network.set_index('episode')['entropy'])

time.sleep(2)
st.rerun()
