import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import json
from utils import get_user_icon, get_assistant_icon
from data_processor import process_data, get_data_info, detect_data_types
from conversation import generate_response

# Set page config
st.set_page_config(
    page_title="Data Analytics Assistant",
    page_icon="📊",
    layout="wide",
)

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "data" not in st.session_state:
    st.session_state.data = None

if "data_info" not in st.session_state:
    st.session_state.data_info = None

if "data_types" not in st.session_state:
    st.session_state.data_types = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Title and description
st.title("Data Analytics Assistant")
st.markdown("""
Upload your dataset and ask questions about it. This assistant can help you:
- Explore and understand your data
- Calculate statistics
- Create visualizations
- Perform correlations and identify patterns
- Make predictions and recommendations
""")

# Create a tabbed interface for data input options
tab1, tab2 = st.tabs(["Example Datasets", "Upload Your Own Data"])

with tab1:
    st.info("Select a built-in dataset to start exploring data analytics capabilities!")
    example_data = st.selectbox(
        "Choose an example dataset",
        ["None", "Sample Sales Data", "Stock Market Data", "Iris Flower Dataset", 
         "Customer Satisfaction Survey", "E-commerce Transactions", "Weather Data"],
        index=0,
        help="Select one of these pre-built datasets to immediately start analyzing data"
    )
    
    # Add dataset descriptions
    if example_data == "Sample Sales Data":
        st.write("📊 **Sample Sales Data**: Contains product sales across different regions with metrics like units sold and customer satisfaction.")
    elif example_data == "Stock Market Data":
        st.write("📈 **Stock Market Data**: Historical stock prices for major tech companies including daily highs, lows, and trading volume.")
    elif example_data == "Iris Flower Dataset":
        st.write("🌸 **Iris Flower Dataset**: Classic dataset containing measurements of iris flowers with species classification.")
    elif example_data == "Customer Satisfaction Survey":
        st.write("📝 **Customer Satisfaction Survey**: Survey responses with ratings across different service aspects, demographic information, and comments.")
    elif example_data == "E-commerce Transactions":
        st.write("🛒 **E-commerce Transactions**: Online shopping transaction data including product categories, prices, customer information, and purchase patterns.")
    elif example_data == "Weather Data":
        st.write("🌦️ **Weather Data**: Daily weather readings from multiple locations including temperature, precipitation, humidity, and wind measurements.")

with tab2:
    st.write("### Paste Your Own Data")
    st.info("You can paste your CSV data directly here to bypass the upload limitations.")
    
    paste_option = st.radio("Choose data input method:", 
                           ["Paste CSV Text", "Enter Comma-separated Data"], 
                           index=0)
    
    if paste_option == "Paste CSV Text":
        pasted_data = st.text_area("Paste your CSV data here (including headers):", 
                                  height=300, 
                                  help="Copy and paste data from CSV file or spreadsheet. Include column headers in the first row.")
        
        has_header = st.checkbox("First row contains column headers", value=True)
        
        if st.button("Process Pasted Data"):
            if pasted_data:
                try:
                    # Store the pasted data in session state
                    st.session_state.pasted_data = pasted_data
                    st.session_state.has_header = has_header
                    # This will trigger the pasted data processing below
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing pasted data: {str(e)}")
    
    else:  # Manual entry
        st.write("Enter your data manually with comma-separated values.")
        
        # Input for column names
        col_names = st.text_input("Enter column names (comma-separated):", 
                                help="Example: Name,Age,Salary")
        
        # Input for data rows
        data_rows = st.text_area("Enter data rows (one row per line, comma-separated values):", 
                               height=250,
                               help="Example:\nJohn,30,50000\nJane,25,60000")
        
        if st.button("Process Manual Data"):
            if col_names and data_rows:
                try:
                    # Combine headers and data to create CSV format
                    combined_data = col_names + "\n" + data_rows
                    # Store in session state
                    st.session_state.pasted_data = combined_data
                    st.session_state.has_header = True
                    # This will trigger the pasted data processing below
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing manual data: {str(e)}")
    
    # Divider for visual separation
    st.markdown("---")
    
    # Keep the original uploader as a fallback, but with clearer warning
    st.warning("⚠️ The standard file uploader below may not work due to environment limitations.")
    uploaded_file = st.file_uploader("Try uploading your dataset (limited functionality)", 
                                    type=["csv", "xlsx", "xls"], 
                                    key="file_upload", 
                                    help="Supported formats: CSV, Excel, but note this may not work in this environment.")

# Process pasted data if available
if hasattr(st.session_state, 'pasted_data') and st.session_state.pasted_data:
    try:
        with st.spinner("Processing pasted data..."):
            # Convert pasted string to DataFrame
            from io import StringIO
            
            # Use the StringIO object to create a file-like object from the string
            data_buffer = StringIO(st.session_state.pasted_data)
            
            # Read the CSV data
            has_header = getattr(st.session_state, 'has_header', True)
            if has_header:
                df = pd.read_csv(data_buffer)
            else:
                df = pd.read_csv(data_buffer, header=None)
                # Assign default column names
                df.columns = [f'Column_{i+1}' for i in range(df.shape[1])]
            
            # Set a default file name
            file_name = "pasted_data.csv"
            
            # Update session state
            st.session_state.data = df
            st.session_state.file_name = file_name
            st.session_state.data_info = get_data_info(df)
            st.session_state.data_types = detect_data_types(df)
            
            # Reset conversation when new file is uploaded
            st.session_state.messages = [
                {"role": "assistant", "content": f"I've loaded your pasted data. The dataset has {df.shape[0]} rows and {df.shape[1]} columns. Ask me anything about your data!"}
            ]
            
            # Show success message
            st.success("Your pasted data was successfully loaded!")
            
            # Clear the pasted data to prevent reloading on page refresh
            st.session_state.pasted_data = None
            
    except Exception as e:
        st.error(f"Error loading pasted data: {str(e)}")
        st.info("Make sure your data is in a valid CSV format with proper delimiters.")
        # Clear the pasted data to allow another attempt
        st.session_state.pasted_data = None

# Process uploaded file or example data
elif uploaded_file is not None:
    try:
        # Get file information
        file_name = uploaded_file.name
        file_extension = file_name.split(".")[-1].lower()
        
        # Show loading message
        with st.spinner(f"Processing {file_name}..."):
            # Save the file to a persistent location
            save_path = os.path.join("data_files", file_name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the data based on file extension
            if file_extension == 'csv':
                df = pd.read_csv(save_path)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(save_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
            
            # Update session state
            st.session_state.data = df
            st.session_state.file_name = file_name
            st.session_state.data_info = get_data_info(df)
            st.session_state.data_types = detect_data_types(df)
            
            # Reset conversation when new file is uploaded
            st.session_state.messages = [
                {"role": "assistant", "content": f"I've loaded your data from '{file_name}'. The dataset has {df.shape[0]} rows and {df.shape[1]} columns. Ask me anything about your data!"}
            ]
            
            # Show success message
            st.success(f"File '{file_name}' successfully loaded!")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Try refreshing the page if you encounter persistent upload issues.")

# Process example data if selected and no file is uploaded
elif example_data != "None":
    try:
        with st.spinner(f"Loading {example_data}..."):
            # Create example datasets
            if example_data == "Sample Sales Data":
                # Create a simple sales dataset
                data = {
                    'Date': pd.date_range(start='2023-01-01', periods=100),
                    'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 100),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
                    'Sales': np.random.randint(100, 1000, 100),
                    'Units': np.random.randint(1, 50, 100),
                    'Customer_Satisfaction': np.random.uniform(1, 5, 100).round(1)
                }
                df = pd.DataFrame(data)
                file_name = "sample_sales_data.csv"
                
            elif example_data == "Stock Market Data":
                # Create a simple stock market dataset
                dates = pd.date_range(start='2022-01-01', periods=252)  # Trading days in a year
                stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
                
                all_data = []
                for stock in stocks:
                    # Starting price between 50 and 500
                    base_price = np.random.uniform(50, 500)
                    # Daily returns with slight positive drift
                    returns = np.random.normal(0.0005, 0.015, len(dates))
                    # Cumulative returns
                    price_path = base_price * np.cumprod(1 + returns)
                    
                    # Create data for this stock
                    stock_data = pd.DataFrame({
                        'Date': dates,
                        'Stock': stock,
                        'Price': price_path,
                        'Volume': np.random.randint(100000, 10000000, len(dates)),
                        'High': price_path * np.random.uniform(1.001, 1.03, len(dates)),
                        'Low': price_path * np.random.uniform(0.97, 0.999, len(dates))
                    })
                    all_data.append(stock_data)
                
                # Combine all stocks
                df = pd.concat(all_data, ignore_index=True)
                file_name = "stock_market_data.csv"
                
            elif example_data == "Iris Flower Dataset":
                # Create a simplified version of the iris dataset
                # Sepal length, Sepal width, Petal length, Petal width
                iris_data = [
                    [5.1, 3.5, 1.4, 0.2, 'setosa'],
                    [4.9, 3.0, 1.4, 0.2, 'setosa'],
                    [4.7, 3.2, 1.3, 0.2, 'setosa'],
                    [4.6, 3.1, 1.5, 0.2, 'setosa'],
                    [5.0, 3.6, 1.4, 0.2, 'setosa'],
                    [5.4, 3.9, 1.7, 0.4, 'setosa'],
                    [4.6, 3.4, 1.4, 0.3, 'setosa'],
                    [5.0, 3.4, 1.5, 0.2, 'setosa'],
                    [4.4, 2.9, 1.4, 0.2, 'setosa'],
                    [4.9, 3.1, 1.5, 0.1, 'setosa'],
                    [7.0, 3.2, 4.7, 1.4, 'versicolor'],
                    [6.4, 3.2, 4.5, 1.5, 'versicolor'],
                    [6.9, 3.1, 4.9, 1.5, 'versicolor'],
                    [5.5, 2.3, 4.0, 1.3, 'versicolor'],
                    [6.5, 2.8, 4.6, 1.5, 'versicolor'],
                    [5.7, 2.8, 4.5, 1.3, 'versicolor'],
                    [6.3, 3.3, 4.7, 1.6, 'versicolor'],
                    [4.9, 2.4, 3.3, 1.0, 'versicolor'],
                    [6.6, 2.9, 4.6, 1.3, 'versicolor'],
                    [5.2, 2.7, 3.9, 1.4, 'versicolor'],
                    [6.3, 3.3, 6.0, 2.5, 'virginica'],
                    [5.8, 2.7, 5.1, 1.9, 'virginica'],
                    [7.1, 3.0, 5.9, 2.1, 'virginica'],
                    [6.3, 2.9, 5.6, 1.8, 'virginica'],
                    [6.5, 3.0, 5.8, 2.2, 'virginica'],
                    [7.6, 3.0, 6.6, 2.1, 'virginica'],
                    [4.9, 2.5, 4.5, 1.7, 'virginica'],
                    [7.3, 2.9, 6.3, 1.8, 'virginica'],
                    [6.7, 2.5, 5.8, 1.8, 'virginica'],
                    [7.2, 3.6, 6.1, 2.5, 'virginica']
                ]
                
                df = pd.DataFrame(iris_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
                file_name = "iris_dataset.csv"
                
            elif example_data == "Customer Satisfaction Survey":
                # Generate customer satisfaction survey data
                num_responses = 150
                
                # Customer demographics
                age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
                genders = ['Male', 'Female', 'Non-binary', 'Prefer not to say']
                membership_levels = ['Bronze', 'Silver', 'Gold', 'Platinum']
                
                # Create the dataset
                data = {
                    'Response_ID': [f'R{i:04d}' for i in range(1, num_responses + 1)],
                    'Date': pd.date_range(start='2023-01-01', end='2023-12-31', periods=num_responses),
                    'Age_Group': np.random.choice(age_groups, num_responses),
                    'Gender': np.random.choice(genders, num_responses, p=[0.48, 0.48, 0.02, 0.02]),
                    'Membership_Level': np.random.choice(membership_levels, num_responses, p=[0.4, 0.3, 0.2, 0.1]),
                    'Product_Rating': np.random.randint(1, 6, num_responses),
                    'Service_Rating': np.random.randint(1, 6, num_responses),
                    'Value_Rating': np.random.randint(1, 6, num_responses),
                    'UX_Rating': np.random.randint(1, 6, num_responses),
                    'Recommend_Score': np.random.randint(0, 11, num_responses),
                    'Years_as_Customer': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20], num_responses),
                }
                
                # Create correlation between satisfaction and recommendation score
                for i in range(num_responses):
                    avg_rating = (data['Product_Rating'][i] + data['Service_Rating'][i] + 
                                 data['Value_Rating'][i] + data['UX_Rating'][i]) / 4.0
                    
                    # NPS score should correlate with but not exactly match ratings
                    base_nps = int(avg_rating * 2)  # Base NPS on a scale of 0-10
                    # Add some noise, but keep within bounds
                    data['Recommend_Score'][i] = min(10, max(0, base_nps + np.random.randint(-2, 3)))
                
                df = pd.DataFrame(data)
                file_name = "customer_satisfaction_survey.csv"
                
            elif example_data == "E-commerce Transactions":
                # Generate e-commerce transaction data
                num_transactions = 200
                
                # Basic data
                customer_ids = [f'CUST{i:04d}' for i in range(1, 51)]  # 50 customers
                product_categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Beauty', 'Toys']
                payment_methods = ['Credit Card', 'PayPal', 'Bank Transfer', 'Gift Card']
                shipping_methods = ['Standard', 'Express', 'Next Day', 'Pickup']
                
                # Create transaction data
                data = {
                    'Transaction_ID': [f'TX{i:05d}' for i in range(1, num_transactions + 1)],
                    'Date': pd.date_range(start='2023-01-01', end='2023-12-31', periods=num_transactions),
                    'Customer_ID': np.random.choice(customer_ids, num_transactions),
                    'Product_Category': np.random.choice(product_categories, num_transactions),
                    'Product_ID': [f'P{np.random.randint(1, 1000):04d}' for _ in range(num_transactions)],
                    'Quantity': np.random.randint(1, 6, num_transactions),
                    'Price': np.random.uniform(10, 500, num_transactions).round(2),
                    'Shipping_Cost': np.random.uniform(0, 50, num_transactions).round(2),
                    'Payment_Method': np.random.choice(payment_methods, num_transactions),
                    'Shipping_Method': np.random.choice(shipping_methods, num_transactions),
                    'Rating': np.random.choice([0, 1, 2, 3, 4, 5], num_transactions, p=[0.3, 0.05, 0.05, 0.1, 0.2, 0.3]),
                    'Return': np.random.choice([True, False], num_transactions, p=[0.1, 0.9])
                }
                
                # Calculate total amount
                data['Total_Amount'] = (data['Price'] * data['Quantity'] + data['Shipping_Cost']).round(2)
                
                # Make shipping costs consistent with shipping method
                for i in range(num_transactions):
                    if data['Shipping_Method'][i] == 'Standard':
                        data['Shipping_Cost'][i] = np.random.uniform(5, 10, 1)[0].round(2)
                    elif data['Shipping_Method'][i] == 'Express':
                        data['Shipping_Cost'][i] = np.random.uniform(10, 20, 1)[0].round(2)
                    elif data['Shipping_Method'][i] == 'Next Day':
                        data['Shipping_Cost'][i] = np.random.uniform(20, 35, 1)[0].round(2)
                    else:  # Pickup
                        data['Shipping_Cost'][i] = 0.0
                        
                    # Recalculate total with updated shipping
                    data['Total_Amount'][i] = (data['Price'][i] * data['Quantity'][i] + data['Shipping_Cost'][i]).round(2)
                
                df = pd.DataFrame(data)
                file_name = "ecommerce_transactions.csv"
                
            elif example_data == "Weather Data":
                # Generate weather data for multiple cities
                cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
                num_days = 365  # One year of data
                start_date = pd.Timestamp('2023-01-01')
                
                all_data = []
                
                for city in cities:
                    # Base temperatures for each city (average for the year)
                    if city == 'New York':
                        base_temp = 15  # Average in Celsius
                    elif city == 'Los Angeles':
                        base_temp = 19
                    elif city == 'Chicago':
                        base_temp = 11
                    elif city == 'Houston':
                        base_temp = 21
                    else:  # Phoenix
                        base_temp = 24
                    
                    # Create seasonal pattern (sine wave)
                    days = np.arange(num_days)
                    seasonal_pattern = 10 * np.sin(2 * np.pi * days / 365)  # +/- 10 degrees seasonal variation
                    
                    # Add random variations
                    daily_variation = np.random.normal(0, 3, num_days)  # Daily random variations
                    
                    # Calculate temperatures
                    temp_max = base_temp + seasonal_pattern + daily_variation + np.random.uniform(2, 5, num_days)
                    temp_min = base_temp + seasonal_pattern + daily_variation - np.random.uniform(2, 5, num_days)
                    temp_avg = (temp_max + temp_min) / 2
                    
                    # Generate dates
                    dates = [start_date + pd.Timedelta(days=i) for i in range(num_days)]
                    
                    # Generate precipitation data (correlated with temperature)
                    precip_prob = 1 / (1 + np.exp(0.3 * (temp_max - base_temp)))  # Logistic function
                    precipitation = np.zeros(num_days)
                    for i in range(num_days):
                        if np.random.random() < precip_prob[i]:
                            precipitation[i] = np.random.exponential(5)  # mm of rain when it rains
                    
                    # Generate humidity data (inversely related to temperature)
                    humidity = 100 - (temp_max - temp_min) * 2 + np.random.normal(0, 8, num_days)
                    humidity = np.clip(humidity, 20, 100)  # Constrain between 20% and 100%
                    
                    # Generate wind speed (somewhat random)
                    wind_speed = np.random.gamma(2, 2, num_days)  # in m/s
                    
                    # Create city data
                    city_data = pd.DataFrame({
                        'Date': dates,
                        'City': city,
                        'Temp_Max_C': temp_max.round(1),
                        'Temp_Min_C': temp_min.round(1),
                        'Temp_Avg_C': temp_avg.round(1),
                        'Precipitation_mm': precipitation.round(1),
                        'Humidity_Pct': humidity.round(0),
                        'Wind_Speed_ms': wind_speed.round(1)
                    })
                    
                    all_data.append(city_data)
                
                # Combine all cities
                df = pd.concat(all_data, ignore_index=True)
                file_name = "weather_data.csv"
            
            # Update session state
            st.session_state.data = df
            st.session_state.file_name = file_name
            st.session_state.data_info = get_data_info(df)
            st.session_state.data_types = detect_data_types(df)
            
            # Reset conversation
            st.session_state.messages = [
                {"role": "assistant", "content": f"I've loaded the {example_data}. The dataset has {df.shape[0]} rows and {df.shape[1]} columns. Ask me anything about this data!"}
            ]
            
            # Show success message
            st.success(f"{example_data} successfully loaded!")
    
    except Exception as e:
        st.error(f"Error loading example data: {str(e)}")
        st.info("Please try selecting a different example dataset.")

# Create chat UI if we have data
if st.session_state.data is not None:
    
    # Display basic data info
    with st.expander("Dataset Overview", expanded=False):
        st.dataframe(st.session_state.data.head(10))
        st.write(f"Shape: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Column Names:")
            st.write(", ".join(st.session_state.data.columns.tolist()))
        
        with col2:
            st.write("Data Types:")
            for col, dtype in zip(st.session_state.data.columns, st.session_state.data.dtypes):
                st.write(f"- {col}: {dtype}")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar=get_user_icon()):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=get_assistant_icon()):
                st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask me about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user", avatar=get_user_icon()):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant", avatar=get_assistant_icon()):
            with st.spinner("Analyzing data..."):
                message_placeholder = st.empty()
                full_response = generate_response(
                    prompt, 
                    st.session_state.messages, 
                    st.session_state.data,
                    st.session_state.data_info,
                    st.session_state.data_types
                )
                message_placeholder.markdown(full_response)
                
                # Display any generated images directly in the UI
                if "image_data" in st.session_state and st.session_state.image_data:
                    st.write("### Visualizations")
                    for key, img_data in st.session_state.image_data.items():
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        
                        # Convert base64 string to image and display it
                        try:
                            image_bytes = base64.b64decode(img_data)
                            image = Image.open(BytesIO(image_bytes))
                            st.image(image, use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not display image: {str(e)}")
                    
                    # Clear the images after displaying
                    st.session_state.image_data = {}
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    # If no data is uploaded yet
    st.info("Please upload a dataset to start the conversation.")
