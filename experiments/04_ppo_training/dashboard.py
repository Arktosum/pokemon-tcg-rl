import json
import os
import time
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Titan PPO Mission Control",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Titan PPO Mission Control")

LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "metrics.jsonl")


def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except Exception:
                    continue
    df = pd.DataFrame(data)
    
    EXPECTED_KEYS = [
        'reward', 'win', 'steps', 'sps', 'duration_sec',
        'pg_loss', 'v_loss', 'entropy', 'explained_variance',
        'kl_divergence', 'grad_norm', 'val_win_rate'
    ]
    for key in EXPECTED_KEYS:
        if key not in df.columns:
            df[key] = None
            
    # Bridge for new decoupled metrics format
    if 'metrics/ep_reward' in df.columns:
        df['reward'] = df['metrics/ep_reward']
        df['win'] = (df['metrics/ep_reward'] > 0).astype(int)
    if 'metrics/ep_length' in df.columns:
        df['steps'] = df['metrics/ep_length']
    if 'metrics/sps' in df.columns:
        df['sps'] = df['metrics/sps']
    if 'metrics/duration_sec' in df.columns:
        df['duration_sec'] = df['metrics/duration_sec']
        
    if 'train/pg_loss' in df.columns:
        df['pg_loss'] = df['train/pg_loss']
    if 'train/v_loss' in df.columns:
        df['v_loss'] = df['train/v_loss']
    if 'train/entropy' in df.columns:
        df['entropy'] = df['train/entropy']
    if 'train/explained_variance' in df.columns:
        df['explained_variance'] = df['train/explained_variance']
    if 'train/kl_divergence' in df.columns:
        df['kl_divergence'] = df['train/kl_divergence']
    if 'train/grad_norm' in df.columns:
        df['grad_norm'] = df['train/grad_norm']
        
    # Forward fill to handle interleaved episode and training logs
    df = df.ffill()
    return df


df = load_data()

if df.empty:
    st.info(
        "Waiting for telemetry data... Launch train_sequential.py to begin"
        " logging."
    )
else:
    recent_df = df.tail(100)
    prev_df = df.iloc[-200:-100] if len(df) >= 200 else pd.DataFrame()

    def get_diff(col_name):
        if prev_df.empty or col_name not in recent_df.columns or col_name not in prev_df.columns:
            return None
        return recent_df[col_name].mean() - prev_df[col_name].mean()

    def render_custom_metric(title, value_str, expected_dir, diff_val, diff_str, is_inverse=False, tooltip=""):
        if diff_val is None:
            actual_html = '<div style="font-size: 13px; color: #a0aec0;">Actual: N/A</div>'
        else:
            if diff_val > 0:
                actual_arrow = "⬆"
                actual_color = "#f56565" if is_inverse else "#48bb78"
                diff_prefix = "+"
            elif diff_val < 0:
                actual_arrow = "⬇"
                actual_color = "#48bb78" if is_inverse else "#f56565"
                diff_prefix = ""
            else:
                actual_arrow = "▬"
                actual_color = "#a0aec0"
                diff_prefix = ""
            actual_html = f'<div style="font-size: 13px; font-weight: bold; color: {actual_color};">Actual: {actual_arrow} ({diff_prefix}{diff_str})</div>'

        exp_arrow = "⬆" if expected_dir == "up" else "⬇"
        if expected_dir == "any":
            exp_arrow = "▬"
        
        html = f"""
        <div style="background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 16px; margin-bottom: 10px; box-shadow: 0 8px 16px rgba(0,0,0,0.2); color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;" title="{tooltip}">
            <div style="font-size: 13px; color: #a0aec0; font-weight: 600; display: flex; justify-content: space-between; text-transform: uppercase; letter-spacing: 0.5px;">{title} <span style="cursor:help; opacity: 0.7;">ⓘ</span></div>
            <div style="font-size: 26px; font-weight: 700; margin: 8px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{value_str}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 12px; color: #cbd5e0;">Expected: <span style="font-size: 14px;">{exp_arrow}</span></div>
                {actual_html}
            </div>
        </div>
        """
        return html

    # Top Metrics Board (Row 1: Performance & Speed)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    ep_diff = int(df["episode"].max()) - (int(prev_df["episode"].max()) if not prev_df.empty else 0)
    col1.markdown(render_custom_metric("Total Episodes", str(int(df["episode"].max())), "up", ep_diff if not prev_df.empty else None, str(ep_diff), False, "Total number of episodes completed."), unsafe_allow_html=True)
    
    win_diff = get_diff("win")
    col2.markdown(render_custom_metric("Recent Win Rate", f"{(recent_df['win'].mean() * 100):.1f}%", "up", win_diff, f"{win_diff * 100:.1f}%" if win_diff is not None else "", False, "Win rate over the last 100 episodes. Higher is better."), unsafe_allow_html=True)
    
    val_win = df.dropna(subset=["val_win_rate"]).tail(10)
    val_prev = df.dropna(subset=["val_win_rate"]).iloc[-20:-10] if len(df.dropna(subset=["val_win_rate"])) >= 20 else pd.DataFrame()
    val_win_str = f"{(val_win['val_win_rate'].mean() * 100):.1f}%" if not val_win.empty else "N/A"
    val_diff = (val_win['val_win_rate'].mean() - val_prev['val_win_rate'].mean()) if not val_prev.empty else None
    col3.markdown(render_custom_metric("Val Win Rate", val_win_str, "up", val_diff, f"{val_diff * 100:.1f}%" if val_diff is not None else "", False, "Validation win rate over the last 10 evaluations."), unsafe_allow_html=True)
    
    step_diff = get_diff("steps")
    col4.markdown(render_custom_metric("Avg Ep Steps", f"{recent_df['steps'].mean():.1f}", "down", step_diff, f"{step_diff:.1f}" if step_diff is not None else "", True, "Average steps per episode. Shorter episodes usually indicate decisive wins or losses."), unsafe_allow_html=True)
    
    sps_diff = get_diff("sps")
    col5.markdown(render_custom_metric("Avg Speed", f"{recent_df['sps'].mean():.1f} /s", "up", sps_diff, f"{sps_diff:.1f}" if sps_diff is not None else "", False, "Steps Per Second. System throughput speed. Higher is better."), unsafe_allow_html=True)
    
    dur_diff = get_diff("duration_sec")
    col6.markdown(render_custom_metric("Avg Duration", f"{recent_df['duration_sec'].mean():.2f}s", "down", dur_diff, f"{dur_diff:.2f}s" if dur_diff is not None else "", True, "Real-world time per episode in seconds. Lower is better."), unsafe_allow_html=True)
    
    # Top Metrics Board (Row 2: Network Health)
    ncol1, ncol2, ncol3, ncol4, ncol5, ncol6 = st.columns(6)
    
    pg_diff = get_diff("pg_loss")
    ncol1.markdown(render_custom_metric("Policy Loss", f"{recent_df['pg_loss'].mean():.4f}", "down", pg_diff, f"{pg_diff:.4f}" if pg_diff is not None else "", True, "Actor network loss. Should trend towards 0. Lower is better."), unsafe_allow_html=True)
    
    v_diff = get_diff("v_loss")
    ncol2.markdown(render_custom_metric("Value Loss", f"{recent_df['v_loss'].mean():.4f}", "down", v_diff, f"{v_diff:.4f}" if v_diff is not None else "", True, "Critic network loss (accuracy of value predictions). Lower is better."), unsafe_allow_html=True)
    
    ent_diff = get_diff("entropy")
    ncol3.markdown(render_custom_metric("Entropy", f"{recent_df['entropy'].mean():.4f}", "down", ent_diff, f"{ent_diff:.4f}" if ent_diff is not None else "", False, "Action randomness/exploration. Should slowly decrease as the agent gains confidence."), unsafe_allow_html=True)
    
    kl_diff = get_diff("kl_divergence")
    ncol4.markdown(render_custom_metric("KL Divergence", f"{recent_df['kl_divergence'].mean():.5f}", "down", kl_diff, f"{kl_diff:.5f}" if kl_diff is not None else "", True, "Change in policy per update. Should remain close to the clip threshold."), unsafe_allow_html=True)
    
    grad_diff = get_diff("grad_norm")
    ncol5.markdown(render_custom_metric("Gradient Norm", f"{recent_df['grad_norm'].mean():.3f}", "down", grad_diff, f"{grad_diff:.3f}" if grad_diff is not None else "", True, "Size of neural network updates. Large spikes indicate instability."), unsafe_allow_html=True)
    
    exp_diff = get_diff("explained_variance")
    ncol6.markdown(render_custom_metric("Exp Variance", f"{recent_df['explained_variance'].mean():.3f}", "up", exp_diff, f"{exp_diff:.3f}" if exp_diff is not None else "", False, "How much variance in the returns is explained by the value function. Should trend towards 1.0."), unsafe_allow_html=True)

    st.divider()

    tab_gameplay, tab_throughput, tab_network = st.tabs(
        ["🎮 Gameplay & Strategy", "⚡ System Throughput", "🧠 Network Health"]
    )

    with tab_gameplay:
        st.subheader("📊 Win Rate & Performance per Opponent")

        if "opponent" in df.columns and "win" in df.columns:
            # Aggregate overall stats per opponent using only the last 100 episodes
            opp_stats = (
                df.tail(100).groupby("opponent")
                .agg(
                    total_games=("win", "count"),
                    total_wins=("win", "sum"),
                    win_rate=("win", "mean"),
                )
                .reset_index()
            )
            opp_stats["win_rate_pct"] = opp_stats["win_rate"] * 100

            # Interactive Altair Bar Chart taking FULL WIDTH
            bar_chart = (
                alt.Chart(opp_stats)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X(
                        "opponent:N",
                        title="Opponent",
                        sort="-y",
                        axis=alt.Axis(labelAngle=-20),
                    ),
                    y=alt.Y(
                        "win_rate_pct:Q",
                        title="Win Rate (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    color=alt.Color("opponent:N", legend=None),
                    tooltip=[
                        alt.Tooltip("opponent:N", title="Opponent"),
                        alt.Tooltip(
                            "win_rate_pct:Q",
                            format=".1f",
                            title="Win Rate (%)",
                        ),
                        alt.Tooltip("total_wins:Q", title="Total Wins"),
                        alt.Tooltip("total_games:Q", title="Total Matches"),
                    ],
                )
                .properties(height=380)
            )

            # Text overlay for win rate percentage on top of bars
            text_overlay = bar_chart.mark_text(
                align="center", baseline="bottom", dy=-5
            ).encode(text=alt.Text("win_rate_pct:Q", format=".1f"))

            st.altair_chart(
                bar_chart + text_overlay, width="stretch"
            )

        st.divider()
        st.subheader("Episode Length Trend (Steps per Game)")
        st.line_chart(df.set_index("episode")["steps"].rolling(100, min_periods=1).mean())

        if "val_win_rate" in df.columns:
            st.subheader("Validation Win Rate Trend")
            val_df = df.dropna(subset=["val_win_rate"])
            if not val_df.empty:
                st.line_chart(val_df.set_index("episode")["val_win_rate"].rolling(10, min_periods=1).mean())

    with tab_throughput:
        st.subheader("Steps Per Second (SPS)")
        st.line_chart(df.set_index("episode")["sps"].rolling(100, min_periods=1).mean())
        
        st.subheader("Episode Duration (Seconds)")
        st.line_chart(df.set_index("episode")["duration_sec"].rolling(100, min_periods=1).mean())

    with tab_network:
        st.subheader("Policy Loss (Actor)")
        st.line_chart(df.set_index("episode")["pg_loss"].rolling(100, min_periods=1).mean())
        
        st.subheader("Value Loss (Critic)")
        st.line_chart(df.set_index("episode")["v_loss"].rolling(100, min_periods=1).mean())
        
        st.subheader("Entropy (Exploration)")
        st.line_chart(df.set_index("episode")["entropy"].rolling(100, min_periods=1).mean())
        
        st.subheader("KL Divergence")
        st.line_chart(df.set_index("episode")["kl_divergence"].rolling(100, min_periods=1).mean())
        
        st.subheader("Gradient Norm")
        st.line_chart(df.set_index("episode")["grad_norm"].rolling(100, min_periods=1).mean())
        
        st.subheader("Explained Variance")
        st.line_chart(df.set_index("episode")["explained_variance"].rolling(100, min_periods=1).mean())

time.sleep(3)
st.rerun()