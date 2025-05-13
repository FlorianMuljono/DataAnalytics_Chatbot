import os
import pandas as pd
import numpy as np
import json
import re
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from utils import plot_to_base64, create_matplotlib_figure, create_plotly_figure

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def generate_system_prompt(data_info, data_types):
    """Generate a system prompt for the OpenAI API based on dataset information"""
    
    system_prompt = """You are an expert data analytics assistant that helps users analyze their datasets.
Your primary goal is to answer questions about data, perform data analysis, visualize data, and provide insights.

You have the following capabilities:
1. Analyze data from pandas DataFrames
2. Create visualizations using matplotlib, seaborn, and plotly
3. Calculate statistics and perform data analysis
4. Make recommendations based on the data

When generating responses, please follow these rules:
- Always provide clear explanations of your analysis
- Include visualizations wherever appropriate
- Format your responses using markdown for readability
- Use code blocks to show python code when relevant
- When generating charts, call the appropriate functions
- Always respond with actionable insights based on the data

IMPORTANT: The user has uploaded a dataset with the following information:"""

    # Add dataset information
    system_prompt += f"\n\nDataset Shape: {data_info['shape'][0]} rows, {data_info['shape'][1]} columns"
    system_prompt += f"\nColumns: {', '.join(data_info['columns'])}"
    
    # Add column data types
    system_prompt += "\n\nColumn Data Types:"
    for col, dtype in data_types.items():
        system_prompt += f"\n- {col}: {dtype}"
    
    # Add information about numerical columns
    if data_info['numeric_stats']:
        system_prompt += "\n\nNumerical Columns Statistics:"
        for col in data_info['numeric_stats'].keys():
            system_prompt += f"\n- {col}"
    
    # Add information about categorical columns
    if data_info['categorical_stats']:
        system_prompt += "\n\nCategorical Columns:"
        for col in data_info['categorical_stats'].keys():
            system_prompt += f"\n- {col} (unique values: {data_info['categorical_stats'][col]['unique_count']})"
    
    # Add information about date columns
    if data_info['date_stats']:
        system_prompt += "\n\nDate Columns:"
        for col, stats in data_info['date_stats'].items():
            system_prompt += f"\n- {col} (range: {stats['min_date']} to {stats['max_date']})"
    
    # Add information about missing values
    system_prompt += "\n\nMissing Values:"
    for col, count in data_info['missing_values'].items():
        if count > 0:
            percentage = data_info['missing_percentage'][col]
            system_prompt += f"\n- {col}: {count} missing values ({percentage:.2f}%)"
    
    # Add additional instructions for visualization
    system_prompt += """

When creating visualizations, you should describe what type of visualization to create and what data to use.
The system will generate the appropriate visualization using matplotlib, seaborn, or plotly.

You can execute code to generate insights. Always provide clear explanations of your analysis.

Remember to maintain context throughout the conversation and refer to previous analyses when appropriate.
"""
    
    return system_prompt

def execute_code(code, df):
    """Safely execute Python code with the DataFrame"""
    # Create a local environment with just the data
    local_env = {"df": df.copy(), "pd": pd, "np": np, 
                 "plt": plt, "sns": sns, "px": px, "go": go,
                 "io": io, "base64": base64}
    
    # Capture output
    output_capture = io.StringIO()
    
    try:
        # Execute the code and capture stdout
        exec(code, local_env, local_env)
        
        # Check if there's a figure in the local environment
        if 'fig' in local_env:
            # Handle matplotlib figure
            if isinstance(local_env['fig'], plt.Figure) or 'matplotlib.figure.Figure' in str(type(local_env['fig'])):
                # Close any existing plots to avoid warnings
                plt.close('all')
                
                # Create a new figure buffer
                buf = io.BytesIO()
                local_env['fig'].savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(local_env['fig'])
                
                return {"type": "image", "data": img_str, "format": "matplotlib"}
            
            # Handle plotly figure
            elif 'plotly.graph_objs' in str(type(local_env['fig'])):
                fig_json = local_env['fig'].to_json()
                return {"type": "plotly", "data": fig_json}
        
        # Check for matplotlib figures generated without assigning to 'fig'
        current_fig = plt.gcf()
        if current_fig.get_axes():
            # There are plots in the current figure
            buf = io.BytesIO()
            current_fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(current_fig)
            
            return {"type": "image", "data": img_str, "format": "matplotlib"}
        
        # Handle DataFrame result
        if 'result' in local_env and isinstance(local_env['result'], pd.DataFrame):
            return {"type": "dataframe", "data": local_env['result'].to_html()}
        
        # Handle numeric or text result
        if 'result' in local_env:
            return {"type": "text", "data": str(local_env['result'])}
        
        # No specific result found, return success message
        return {"type": "text", "data": "Code executed successfully, but no specific result was returned."}
    
    except Exception as e:
        # Return error message
        return {"type": "error", "data": str(e)}

def process_code_blocks(response_text, df):
    """Process code blocks in the response and execute them"""
    # Import re at the function level to ensure it's available
    import re
    
    # Improved regex pattern to find Python code blocks with better handling of various formats
    pattern = r'```python\s*(.*?)\s*```'
    
    # Use re.DOTALL to match across newlines
    code_blocks = re.findall(pattern, response_text, re.DOTALL)
    
    new_response = response_text
    
    for i, code in enumerate(code_blocks):
        # Strip extra whitespace for consistency
        clean_code = code.strip()
        
        # Execute the code
        result = execute_code(clean_code, df)
        
        # Original code block pattern to replace (handle different whitespace formats)
        original_block = f"```python\n{code}\n```"
        if original_block not in new_response:
            # Try alternative formats
            original_block = f"```python\n{code}```"
            if original_block not in new_response:
                original_block = f"```python{code}```"
                if original_block not in new_response:
                    # Last resort - use regex to find the exact block
                    pattern = re.compile(r'```python\s*' + re.escape(code) + r'\s*```', re.DOTALL)
                    match = pattern.search(new_response)
                    if match:
                        original_block = match.group(0)
        
        # Create replacement text with original code and results
        replacement = f"```python\n{clean_code}\n```"
        
        # Add appropriate result based on type
        if result["type"] == "image":
            # Store the image data in the session state for Streamlit to render directly
            image_key = f"image_{hash(code)}"
            if "image_data" not in st.session_state:
                st.session_state.image_data = {}
            
            st.session_state.image_data[image_key] = result["data"]
            
            # Use markdown that tells users an image should be here
            img_markdown = f"\n\n**[Visualization: A chart has been generated]**\n\n"
            replacement += f"\n\n{img_markdown}"
        
        elif result["type"] == "plotly":
            # Handle Plotly figures
            placeholder = f"[PLOTLY_FIGURE_{i}]"
            replacement += f"\n\n{placeholder}"
            
            # Store the Plotly figure data in session state
            if "plotly_figures" not in st.session_state:
                st.session_state.plotly_figures = {}
            st.session_state.plotly_figures[placeholder] = result["data"]
        
        elif result["type"] == "dataframe":
            replacement += f"\n\n{result['data']}"
        
        elif result["type"] == "error":
            replacement += f"\n\n**Error:** {result['data']}"
        
        else:
            replacement += f"\n\n**Result:** {result['data']}"
        
        # Replace the code block with the new content
        new_response = new_response.replace(original_block, replacement)
    
    return new_response

def generate_response(prompt, messages, df, data_info, data_types):
    """Generate a response using the OpenAI API and process code blocks"""
    # Generate system prompt based on dataset info
    system_prompt = generate_system_prompt(data_info, data_types)
    
    # Prepare conversation history
    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add message history (excluding system messages)
    for message in messages:
        if message["role"] != "system":
            conversation.append(message)
    
    # Add the new user prompt
    conversation.append({"role": "user", "content": prompt})
    
    try:
        # Call OpenAI API with better error handling
        response = openai.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            messages=conversation,
            temperature=0.5,
            max_tokens=2000
        )
        
        # Get the response text
        response_text = response.choices[0].message.content
        
        # Process code blocks in the response
        processed_response = process_code_blocks(response_text, df)
        
        # Note: We'll handle Plotly figures differently
        # Instead of trying to display them here, we'll keep the placeholders
        # and display a note about the plots
        processed_response = processed_response.replace(
            "[PLOTLY_FIGURE_", 
            "**Note:** Plotly figure would be displayed here. "
        )
        
        return processed_response
    
    except Exception as e:
        # Return error message
        return f"I encountered an error while generating a response: {str(e)}"
