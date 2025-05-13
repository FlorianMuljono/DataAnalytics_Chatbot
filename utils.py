import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def get_user_icon():
    """Return the user icon for chat interface"""
    return "👤"

def get_assistant_icon():
    """Return the assistant icon for chat interface"""
    return "🤖"

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in markdown"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def create_matplotlib_figure(plot_type, data, x=None, y=None, title=None, xlabel=None, ylabel=None, **kwargs):
    """Create and return a matplotlib figure based on the specified plot type"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'bar':
        if x is not None and y is not None:
            sns.barplot(x=x, y=y, data=data, ax=ax)
    elif plot_type == 'line':
        if x is not None and y is not None:
            sns.lineplot(x=x, y=y, data=data, ax=ax)
    elif plot_type == 'scatter':
        if x is not None and y is not None:
            sns.scatterplot(x=x, y=y, data=data, ax=ax)
    elif plot_type == 'histogram':
        if x is not None:
            sns.histplot(data[x], ax=ax)
    elif plot_type == 'boxplot':
        if x is not None:
            sns.boxplot(x=x, data=data, ax=ax)
        elif y is not None:
            sns.boxplot(y=y, data=data, ax=ax)
    elif plot_type == 'heatmap':
        if isinstance(data, pd.DataFrame):
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    elif plot_type == 'pie':
        if x is not None and y is not None:
            ax.pie(data[y], labels=data[x], autopct='%1.1f%%')
            ax.axis('equal')
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    return fig

def create_plotly_figure(plot_type, data, x=None, y=None, title=None, xlabel=None, ylabel=None, **kwargs):
    """Create and return a plotly figure based on the specified plot type"""
    
    fig = None
    
    if plot_type == 'bar':
        if x is not None and y is not None:
            fig = px.bar(data, x=x, y=y, title=title)
    elif plot_type == 'line':
        if x is not None and y is not None:
            fig = px.line(data, x=x, y=y, title=title)
    elif plot_type == 'scatter':
        if x is not None and y is not None:
            fig = px.scatter(data, x=x, y=y, title=title)
    elif plot_type == 'histogram':
        if x is not None:
            fig = px.histogram(data, x=x, title=title)
    elif plot_type == 'box':
        if x is not None:
            fig = px.box(data, x=x, title=title)
        elif y is not None:
            fig = px.box(data, y=y, title=title)
    elif plot_type == 'heatmap':
        if isinstance(data, pd.DataFrame):
            fig = px.imshow(data.corr(), text_auto=True, title=title)
    elif plot_type == 'pie':
        if x is not None and y is not None:
            fig = px.pie(data, names=x, values=y, title=title)
    
    if fig:
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel
        )
    
    return fig

def display_chart(chart_func, data, st_chart_func=None, **kwargs):
    """Generic function to display a chart in Streamlit"""
    fig = chart_func(data, **kwargs)
    if st_chart_func:
        st_chart_func(fig, use_container_width=True)
    else:
        st.pyplot(fig)
    
    return fig
