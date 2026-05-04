"""
Advanced Streamlit web app for voice similarity search
với visualizations và feature analysis
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

from src.search.similarity_search import VoiceSimilaritySearch
from src.feature_extraction.extractor import AudioFeatureExtractor
from src.utils.audio_utils import save_audio, preprocess_audio
from src.evaluation.retrieval_evaluator import run_retrieval_evaluation
import os

# Page config
st.set_page_config(
    page_title="Voice Similarity Search - Advanced",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .similarity-score {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Feature names
FEATURE_NAMES = [
    *[f"MFCC_{i}_mean" for i in range(1, 14)],
    *[f"MFCC_{i}_std" for i in range(1, 14)],
    "Pitch_mean", "Pitch_std", "Pitch_min", "Pitch_max",
    "Centroid_mean", "Centroid_std",
    "Rolloff_mean", "Rolloff_std",
    "Bandwidth_mean", "Bandwidth_std",
    "ZCR_mean", "ZCR_std",
    "RMS_mean", "RMS_std",
    *[f"Chroma_{i}" for i in range(12)]
]

FEATURE_GROUPS = {
    "MFCC (Timbre)": list(range(0, 26)),
    "Pitch": list(range(26, 30)),
    "Spectral": list(range(30, 36)),
    "Temporal": list(range(36, 40)),
    "Chroma": list(range(40, 52))
}

# Initialize
@st.cache_resource
def load_search_system():
    try:
        return VoiceSimilaritySearch(), AudioFeatureExtractor(), True, None
    except Exception as e:
        return None, None, False, str(e)

search_system, feature_extractor, system_ready, error_msg = load_search_system()

# Header
st.markdown('<div class="main-header">🎤 Voice Similarity Search - Advanced Analytics</div>', unsafe_allow_html=True)
st.markdown("Phân tích chi tiết 52 đặc trưng âm thanh và trực quan hóa không gian vector")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    if system_ready:
        stats = search_system.get_system_stats()
        st.success("✅ Hệ thống sẵn sàng")
        st.metric("Số giọng nói", stats.get('total_vectors', 0))
        st.metric("Chiều đặc trưng", stats.get('feature_dimension', 0))
        st.info(f"**Index:** {stats.get('index_type', 'N/A')}")
    else:
        st.error("❌ Hệ thống chưa sẵn sàng")
        st.code(error_msg if error_msg else "Build database first")
    
    st.markdown("---")
    
    top_k = st.slider("📊 Số kết quả", 1, 10, 5)
    show_waveform = st.checkbox("📈 Dạng sóng", True)
    show_spectrogram = st.checkbox("🌈 Spectrogram", True)
    show_features = st.checkbox("🔬 Chi tiết features", True)
    show_vector_viz = st.checkbox("🎯 Vector visualization", True)

    st.markdown("---")
    st.subheader("🧪 Retrieval Evaluation")
    eval_top_k = st.slider("Top-K đánh giá", 1, 10, 5, key="eval_top_k")
    if st.button("Run evaluation", use_container_width=True):
        with st.spinner("Đang chạy đánh giá trên data/query_processed..."):
            try:
                eval_summary = run_retrieval_evaluation(
                    query_dir="data/query_processed",
                    metadata_db_path="database/metadata.db",
                    top_k=eval_top_k,
                    output_dir="reports/retrieval",
                )
                st.session_state["eval_summary"] = eval_summary
                st.success("Đánh giá hoàn tất")
            except Exception as e:
                st.error(f"Đánh giá thất bại: {e}")

    if "eval_summary" in st.session_state:
        s = st.session_state["eval_summary"]
        st.metric("Query files", s.get("num_query_files", 0))
        st.metric(f"Hit@{s.get('top_k', eval_top_k)}", f"{s.get('hit_rate_at_k', 0.0):.3f}")
        st.metric("Mean similarity", f"{s.get('mean_similarity_percent', 0.0):.2f}%")
    
    st.markdown("---")
    st.markdown("""
    ### 📝 Hướng dẫn
    1. Upload audio (WAV/MP3/FLAC)
    2. Xem 5 tabs phân tích:
       - 📊 **Results**: Top matches
       - 🔬 **Features**: 52 features
       - 📈 **Comparison**: Feature comparison
       - 🎯 **Vectors**: PCA/t-SNE plot
       - 💡 **Insights**: Giải thích
    """)

# Main
if not system_ready:
    st.error("🚫 Hệ thống chưa sẵn sàng")
    st.stop()

st.subheader("🔎 Chọn nguồn query")
query_mode = st.radio(
    "Nguồn audio query",
    ["Chọn từ danh sách test", "Upload từ máy"],
    horizontal=True,
)

query_audio_path = None
temp_path = None

if query_mode == "Chọn từ danh sách test":
    query_dir = Path("data/query_processed")
    query_files = sorted(query_dir.glob("*.wav"))
    if query_files:
        selected_query = st.selectbox(
            "Danh sách file test",
            options=query_files,
            format_func=lambda p: p.name,
        )
        if st.button("🔍 Tìm kiếm", type="primary"):
            query_audio_path = str(selected_query)
    else:
        st.warning("Không tìm thấy file test trong data/query_processed")
else:
    uploaded_file = st.file_uploader(
        "📤 Upload audio file",
        type=['wav', 'mp3', 'flac'],
        help="Upload audio để tìm giọng tương đồng"
    )
    if uploaded_file is not None and st.button("🔍 Tìm kiếm", type="primary"):
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "query.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        query_audio_path = str(temp_path)

if query_audio_path:
    
    # Load audio
    y_query, sr_query = librosa.load(query_audio_path)
    
    # Extract query features
    with st.spinner('🔍 Đang phân tích...'):
        query_audio_processed = preprocess_audio(query_audio_path)
        query_features = feature_extractor.extract_all_features(query_audio_processed)
        results = search_system.search_similar(query_audio_path, top_k=top_k)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Kết quả",
        "🔬 Features Query",
        "📈 So sánh Features",
        "🎯 Vector Space",
        "💡 Insights"
    ])
    
    # TAB 1: Results
    with tab1:
        st.subheader("🎵 Query Audio")
        col_q1, col_q2 = st.columns([1, 2])
        
        with col_q1:
            st.audio(query_audio_path)
            st.metric("Duration", f"{len(y_query)/sr_query:.2f}s")
            st.metric("Sample Rate", f"{sr_query} Hz")
        
        with col_q2:
            if show_waveform:
                fig_q, ax_q = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(y_query, sr=sr_query, ax=ax_q, color='#667eea')
                ax_q.set_title("Waveform - Query")
                st.pyplot(fig_q)
                plt.close()
        
        st.markdown("---")
        st.subheader(f"🔍 Top {top_k} Similar Voices")
        
        for rank, (file_path, similarity, cosine) in enumerate(results, 1):
            with st.expander(f"#{rank} | {Path(file_path).name} | {similarity:.1f}%", expanded=(rank==1)):
                result_meta = search_system.get_metadata(file_path) or {}
                col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
                
                with col_r1:
                    st.metric("Similarity", f"{similarity:.2f}%")
                    st.metric("Cosine", f"{cosine:.4f}")
                    st.markdown(f"**Voice:** {result_meta.get('voice_name') or 'N/A'}")
                    st.markdown(f"**Video ID:** {result_meta.get('source_video_id') or 'N/A'}")
                    if result_meta.get('source_url'):
                        st.markdown(f"**Source URL:** {result_meta.get('source_url')}")
                
                with col_r2:
                    if os.path.exists(file_path):
                        st.audio(file_path)
                
                with col_r3:
                    if os.path.exists(file_path) and show_waveform:
                        y_r, sr_r = librosa.load(file_path)
                        fig_r, ax_r = plt.subplots(figsize=(6, 2))
                        librosa.display.waveshow(y_r, sr=sr_r, ax=ax_r, color='#764ba2')
                        ax_r.set_title(f"Waveform #{rank}")
                        st.pyplot(fig_r)
                        plt.close()
    
    # TAB 2: Query Features
    with tab2:
        st.subheader("🔬 52 Features của Query Audio")
        
        # Feature table
        df_features = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Value': query_features
        })
        
        # Group by category
        for group_name, indices in FEATURE_GROUPS.items():
            with st.expander(f"📌 {group_name} ({len(indices)} features)", expanded=True):
                group_df = df_features.iloc[indices]
                
                # Bar chart
                fig_bar = px.bar(
                    group_df,
                    x='Feature',
                    y='Value',
                    title=f"{group_name} Values",
                    color='Value',
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Table
                st.dataframe(group_df, use_container_width=True)
    
    # TAB 3: Feature Comparison
    with tab3:
        st.subheader("📈 So sánh Features: Query vs Top Results")
        
        if results:
            # Load top result features
            top_file = results[0][0]
            if os.path.exists(top_file):
                top_features = feature_extractor.extract_from_file(top_file)
                
                # Comparison by group
                for group_name, indices in FEATURE_GROUPS.items():
                    with st.expander(f"📊 {group_name}", expanded=(group_name=="Pitch")):
                        query_group = query_features[indices]
                        top_group = top_features[indices]
                        
                        df_compare = pd.DataFrame({
                            'Feature': [FEATURE_NAMES[i] for i in indices],
                            'Query': query_group,
                            'Top Match': top_group,
                            'Difference': np.abs(query_group - top_group)
                        })
                        
                        # Line chart comparison
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=df_compare['Feature'],
                            y=df_compare['Query'],
                            mode='lines+markers',
                            name='Query',
                            line=dict(color='#667eea', width=2)
                        ))
                        fig_compare.add_trace(go.Scatter(
                            x=df_compare['Feature'],
                            y=df_compare['Top Match'],
                            mode='lines+markers',
                            name='Top Match',
                            line=dict(color='#764ba2', width=2)
                        ))
                        fig_compare.update_layout(
                            title=f"{group_name} Comparison",
                            xaxis_title="Feature",
                            yaxis_title="Value",
                            height=400
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                        
                        # Difference heatmap
                        col_c1, col_c2 = st.columns([2, 1])
                        with col_c1:
                            st.dataframe(df_compare, use_container_width=True)
                        with col_c2:
                            st.metric("Mean Difference", f"{df_compare['Difference'].mean():.4f}")
                            st.metric("Max Difference", f"{df_compare['Difference'].max():.4f}")
    
    # TAB 4: Vector Visualization
    with tab4:
        st.subheader("🎯 Vector Space Visualization")
        
        if show_vector_viz:
            # Load database features
            db_features = np.load('database/features.npy')
            
            with st.spinner('Computing PCA...'):
                # PCA
                pca = PCA(n_components=2)
                db_pca = pca.fit_transform(db_features)
                query_pca = pca.transform(query_features.reshape(1, -1))
                
                # Get top results PCA
                top_indices = []
                for file_path, _, _ in results[:5]:
                    # Find index in feature array by file path
                    row = search_system.get_metadata(file_path)
                    if row and row.get('vector_idx') is not None:
                        top_indices.append(int(row['vector_idx']))
                
                # Plot PCA
                fig_pca = go.Figure()
                
                # Database points
                fig_pca.add_trace(go.Scatter(
                    x=db_pca[:, 0],
                    y=db_pca[:, 1],
                    mode='markers',
                    name='Database',
                    marker=dict(size=5, color='lightgray', opacity=0.5)
                ))
                
                # Top results
                if top_indices:
                    top_pca = db_pca[top_indices]
                    fig_pca.add_trace(go.Scatter(
                        x=top_pca[:, 0],
                        y=top_pca[:, 1],
                        mode='markers',
                        name='Top Matches',
                        marker=dict(size=12, color='orange', symbol='star')
                    ))
                
                # Query point
                fig_pca.add_trace(go.Scatter(
                    x=query_pca[:, 0],
                    y=query_pca[:, 1],
                    mode='markers',
                    name='Query',
                    marker=dict(size=15, color='red', symbol='diamond')
                ))
                
                fig_pca.update_layout(
                    title=f"PCA Projection (Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                    height=600
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                
                st.info("🔴 **Query** | ⭐ **Top Matches** | ⚪ **Database**")
    
    # TAB 5: Insights
    with tab5:
        st.subheader("💡 Phân tích & Giải thích")
        
        if results:
            top_file, top_sim, top_cosine = results[0]
            
            st.markdown(f"""
            ### 🎯 Kết quả tốt nhất
            
            **File:** `{Path(top_file).name}`  
            **Similarity:** {top_sim:.2f}% (Cosine: {top_cosine:.4f})
            
            ### 📊 Phân tích độ tương đồng:
            
            {'🟢 **Rất cao** (>95%)' if top_sim > 95 else '🟡 **Cao** (85-95%)' if top_sim > 85 else '🟠 **Trung bình** (70-85%)' if top_sim > 70 else '🔴 **Thấp** (<70%)'}
            
            """)
            
            # Feature contribution
            if os.path.exists(top_file):
                top_features = feature_extractor.extract_from_file(top_file)
                
                # Calculate feature differences
                diffs = np.abs(query_features - top_features)
                contribution = 1 - (diffs / (np.abs(query_features) + np.abs(top_features) + 1e-8))
                
                # Top contributing features
                top_contrib_idx = np.argsort(contribution)[::-1][:10]
                
                st.markdown("### 🔝 Top 10 Features đóng góp vào similarity:")
                
                contrib_df = pd.DataFrame({
                    'Feature': [FEATURE_NAMES[i] for i in top_contrib_idx],
                    'Contribution': contribution[top_contrib_idx] * 100,
                    'Query Value': query_features[top_contrib_idx],
                    'Match Value': top_features[top_contrib_idx]
                })
                
                fig_contrib = px.bar(
                    contrib_df,
                    x='Feature',
                    y='Contribution',
                    title="Feature Contribution to Similarity",
                    color='Contribution',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                st.dataframe(contrib_df, use_container_width=True)
                
                st.markdown("""
                ### 📖 Giải thích:
                
                **Cosine Similarity** đo góc giữa 2 vectors trong không gian 52 chiều:
                - **1.0 (100%)**: Vectors giống hệt nhau
                - **0.95-1.0 (95-100%)**: Rất tương đồng (cùng speaker)
                - **0.85-0.95 (85-95%)**: Tương đồng cao (giọng giống nhau)
                - **< 0.85 (< 85%)**: Khác biệt đáng kể
                
                **52 Features bao gồm:**
                - 26 MFCC: Âm sắc giọng nói
                - 4 Pitch: Cao độ (F0)
                - 6 Spectral: Brightness, tone
                - 4 Temporal: Energy, rhythm
                - 12 Chroma: Harmonics
                """)
    
    # Cleanup
    if temp_path is not None and temp_path.exists():
        temp_path.unlink()

else:
    st.info("👆 Chọn file test hoặc upload file, sau đó nhấn 'Tìm kiếm'")
    
    if system_ready:
        st.markdown("### 📈 Database Statistics")
        stats = search_system.get_system_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Voices", stats['total_vectors'])
        col2.metric("Features", stats['feature_dimension'])
        col3.metric("Index Type", stats['index_type'])

    if "eval_summary" in st.session_state:
        st.markdown("### 🧪 Retrieval Evaluation Summary")
        s = st.session_state["eval_summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", s.get("num_query_files", 0))
        c2.metric(f"Hit@{s.get('top_k', 5)}", f"{s.get('hit_rate_at_k', 0.0):.3f}")
        c3.metric("MRR", f"{s.get('mean_mrr', 0.0):.3f}")
        c4.metric("Mean sim", f"{s.get('mean_similarity_percent', 0.0):.2f}%")

        st.markdown("**Mean similarity by rank**")
        rank_df = pd.DataFrame(
            {
                "rank": list(s.get("mean_similarity_by_rank", {}).keys()),
                "mean_similarity_percent": list(s.get("mean_similarity_by_rank", {}).values()),
            }
        )
        if not rank_df.empty:
            st.dataframe(rank_df, use_container_width=True)

        outputs = s.get("outputs", {})
        if outputs.get("summary_json") and os.path.exists(outputs["summary_json"]):
            with open(outputs["summary_json"], "r", encoding="utf-8") as f:
                st.download_button(
                    "Download summary.json",
                    data=f.read(),
                    file_name="retrieval_summary.json",
                    mime="application/json",
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>Advanced Voice Similarity Search | SQLite + Cosine Similarity | 52D Feature Space</p>
</div>
""", unsafe_allow_html=True)
