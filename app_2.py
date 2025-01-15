import streamlit as st
import pandas as pd
import re


# Age band dictionary
age_band_dict = {
    "A": "before 1900", "B": "1900-1929", "C": "1930-1949", "D": "1950-1966",
    "E": "1967-1975", "F": "1976-1982", "G": "1983-1990", "H": "1991-1995",
    "I": "1996-2002", "J": "2003-2006", "K": "2007-2011", "L": "2012-2022", "M": "2023 onwards"
}

# Wall U-values dictionary
wall_u_values_england = {
    "Cavity as built": {"A": 1.5, "B": 1.5, "C": 1.5, "D": 1.5, "E": 1.5, "F": 1.0},
    "Filled cavity": {"A": 0.7, "B": 0.7, "C": 0.7, "D": 0.7, "E": 0.7}
}

# Recommendations list
recommendations_list = [
    "Air or ground source heat pump",
    "Double glazed windows",
    "Draught proofing of windows and doors",
    "Cavity wall insulation on its own",
    "Flat roof insulation",
    "Heating controls for wet central heating",
    "Loft insulation at ceiling level",
    "Low energy lights",
    "Roof room insulation",
    "Solid wall insulation (external)",
    "Solid wall insulation (internal)",
    "External Wall insulation on system build & Timber frame walls"
]

# Function to append recommendations
def append_recommendation(current, new):
    return f"{current}, {new}" if current else new

def process_file(file, selected_recommendations, target_score):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None

    # Standardize BUILT_FORM field
    df['BUILT_FORM'] = df['BUILT_FORM'].str.replace('Enclosed ', '', regex=False)
    df['MULTI_GLAZE_PROPORTION'] = df['MULTI_GLAZE_PROPORTION'].astype(str)
    df['MULTI_GLAZE_PROPORTION'] = df['MULTI_GLAZE_PROPORTION'].replace(r'\xa0', '', regex=True).str.strip()

    # Initialize the RECOMMENDATION column
    df['RECOMMENDATION'] = ""

    # Apply selected recommendation logic here
    for recommendation in selected_recommendations:
        if recommendation == "Air or ground source heat pump":
            condition = (
                (df['PROPERTY_TYPE'].str.lower().isin(['house', 'bungalow'])) & 
                (df['CURRENT_ENERGY_RATING'].str.upper().isin(['F', 'G']))
            )
            df.loc[condition, 'RECOMMENDATION'] = df.loc[condition, 'RECOMMENDATION'].apply(
                lambda rec: append_recommendation(rec, recommendation)
            )

        if recommendation in ["Double glazed windows"]:
            df['MULTI_GLAZE_PROPORTION'] = pd.to_numeric(df['MULTI_GLAZE_PROPORTION'], errors='coerce').fillna(0).astype('int64')
            condition = (df['GLAZED_TYPE'].str.lower() == 'single glazing')
            df.loc[condition, 'RECOMMENDATION'] = df.loc[condition, 'RECOMMENDATION'].apply(
                lambda rec: append_recommendation(rec, recommendation)
            )

        if recommendation in ["Draught proofing of windows and doors"]:
            df['MULTI_GLAZE_PROPORTION'] = pd.to_numeric(df['MULTI_GLAZE_PROPORTION'], errors='coerce').fillna(0).astype('int64')
            condition = (df['GLAZED_TYPE'].str.lower() == 'single glazing')
            df.loc[condition, 'RECOMMENDATION'] = df.loc[condition, 'RECOMMENDATION'].apply(
                lambda rec: append_recommendation(rec, recommendation)
            )

        if recommendation == "Cavity wall insulation on its own":
            cavity_wall_direct = [
                "Cavity Wall", "Cavity wall,no insulation (assumed)", "Cavity wall, insulated (assumed)"
            ]
            cavity_wall_as_built = [
                "Cavity wall, as built, insulated (assumed)",
                "Cavity wall, as built, no insulation (assumed)",
                "Cavity wall, as built, partial insulation (assumed)"
            ]
            cavity_wall_filled = ["Cavity wall, filled cavity"]

            condition_direct = df['WALLS_DESCRIPTION'].isin(cavity_wall_direct)
            df.loc[condition_direct, 'RECOMMENDATION'] = df.loc[condition_direct, 'RECOMMENDATION'].apply(
                lambda rec: append_recommendation(rec, recommendation)
            )

            for index, row in df.iterrows():
                wall_description = row['WALLS_DESCRIPTION']
                construction_age_band = str(row['CONSTRUCTION_AGE_BAND'])

                if wall_description in cavity_wall_as_built:
                    age_band = next((k for k, v in age_band_dict.items() if construction_age_band in v), None)
                    if age_band and wall_u_values_england["Cavity as built"].get(age_band, 0) >= 0.7:
                        df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

                elif wall_description in cavity_wall_filled:
                    age_band = next((k for k, v in age_band_dict.items() if construction_age_band in v), None)
                    if age_band and wall_u_values_england["Filled cavity"].get(age_band, 0) >= 0.7:
                        df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "Flat roof insulation":
            flat_roof_values = [
                "Flat", "Flat,", "Flat, insulated (assumed)", "Flat, limited insulation", 
                "Flat, limited insulation (assumed)", "Flat, no insulation", "Flat, no insulation (assumed)"
            ]

            for index, row in df.iterrows():
                roof_description = row['ROOF_DESCRIPTION']
                construction_age_band = row['CONSTRUCTION_AGE_BAND']

                if roof_description in flat_roof_values:
                    age_band = next((k for k, v in age_band_dict.items() if construction_age_band in v), None)
                    if age_band and age_band != "M":
                        df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "Heating controls for wet central heating":
            df['MAIN_HEATING_CONTROLS'] = df['MAIN_HEATING_CONTROLS'].replace(r'\xa0', '', regex=True).str.strip()
            df['MAIN_HEATING_CONTROLS'] = pd.to_numeric(df['MAIN_HEATING_CONTROLS'], errors='coerce').fillna(0).astype('int32')
            df['MAINHEAT_DESCRIPTION'] = df['MAINHEAT_DESCRIPTION'].fillna('').astype(str).str.lower().str.strip()
            heating_control_codes = ["2101", "2102", "2103", "2104", "2105", "2106"]

            for index, row in df.iterrows():
                main_heating_controls = str(row['MAIN_HEATING_CONTROLS'])
                mainheat_description = row['MAINHEAT_DESCRIPTION']

                if main_heating_controls in heating_control_codes and "boiler and radiators" in mainheat_description:
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)
                elif "boiler" in mainheat_description and "underfloor heating" in mainheat_description:
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)
                elif "heat pump, radiators" in mainheat_description or "underfloor" in mainheat_description:
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "Loft insulation at ceiling level":
            for index, row in df.iterrows():
                roof_description = str(row['ROOF_DESCRIPTION']).lower()

                if "pitched, no insulation" in roof_description:
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)
                elif "pitched" in roof_description and "loft insulation" in roof_description:
                    match = re.search(r'(\d+)\s?mm', roof_description)
                    if match:
                        insulation_value = int(match.group(1))
                        if insulation_value < 400:
                            df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "Low energy lights":
            for index, row in df.iterrows():
                light_description = str(row['LIGHTING_DESCRIPTION']).lower()

                match = re.search(r'(\d+)%', light_description)
                if match:
                    percentage = int(match.group(1))
                    if percentage < 100:
                        df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)
                elif "no low energy lighting" in light_description:
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "Roof room insulation":
            roof_room_direct = [
                "roof room(s), insulated (assumed)",
                "roof room(s), limited insulation (assumed)",
                "roof room(s), no insulation (assumed)"
            ]
            roof_room_conditional = [
                "roof room(s), ceiling insulated",
                "roof room(s), insulated"
            ]

            for index, row in df.iterrows():
                roof_description = str(row['ROOF_DESCRIPTION']).lower()
                construction_age_band = row['CONSTRUCTION_AGE_BAND']

                if any(pattern in roof_description for pattern in roof_room_direct):
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)
                elif any(pattern in roof_description for pattern in roof_room_conditional):
                    age_band = next((k for k, v in age_band_dict.items() if construction_age_band in v), None)
                    if age_band in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                        df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "Solid wall insulation (external)":
            solid_brick_patterns = [
                "solid brick, as built, no insulation (assumed)",
                "solid brick, as built, insulated (assumed)",
                "solid brick, as built, partial insulation (assumed)",
                "solid brick, as built"
            ]

            for index, row in df.iterrows():
                walls_description = str(row['WALLS_DESCRIPTION']).lower()
                construction_age_band = str(row.get('CONSTRUCTION_AGE_BAND', ''))

                if any(pattern in walls_description for pattern in solid_brick_patterns):
                    age_band = next((k for k, v in age_band_dict.items() if construction_age_band in v), None)
                    if age_band in ['A', 'B', 'C', 'D', 'E', 'F']:
                        df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "Solid wall insulation (internal)":
            for index, row in df.iterrows():
                walls_description = str(row['WALLS_DESCRIPTION']).lower()
                construction_age_band = str(row.get('CONSTRUCTION_AGE_BAND', ''))
                inside_polygon = str(row.get('INSIDE_POLYGON', '')).lower()

                if (
                    any(pattern in walls_description for pattern in solid_brick_patterns) and
                    (next((k for k, v in age_band_dict.items() if construction_age_band in v), None) in ['A', 'B', 'C', 'D', 'E', 'F'])
                ) or inside_polygon == 'yes':
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

        if recommendation == "External Wall insulation on system build & Timber frame walls":
            timber_frame_patterns = [
                "timber frame, as built, insulated (assumed)",
                "timber frame, as built, no insulation (assumed)",
                "timber frame, as built, partial insulation (assumed)"
            ]
            system_built_patterns = [
                "system built, as built, insulated (assumed)",
                "system built, as built, no insulation (assumed)",
                "system built, as built, partial insulation (assumed)"
            ]

            for index, row in df.iterrows():
                walls_description = str(row['WALLS_DESCRIPTION']).lower()

                if (
                    any(pattern in walls_description for pattern in timber_frame_patterns) or
                    any(pattern in walls_description for pattern in system_built_patterns)
                ):
                    df.at[index, 'RECOMMENDATION'] = append_recommendation(df.at[index, 'RECOMMENDATION'], recommendation)

    # Clean up trailing commas in RECOMMENDATION column
    df['RECOMMENDATION'] = df['RECOMMENDATION'].str.rstrip(', ')
    
        # Scores and priority tables
    measure_scores = {
        "Loft insulation at ceiling level": 2,
        "Low energy lights": 1,
        "Draught proofing of windows and doors": 3,
        "Heating controls for wet central heating": 4,
        "Cavity wall insulation on its own": 6,
        "Double glazed windows": 4,
        "Flat roof insulation": 3,
        "Roof room insulation": 6,
        "Solid wall insulation (external)": 5,
        "Solid wall insulation (internal)": 5,
        "External Wall insulation on system build & Timber frame walls": 5,
        "Air or ground source heat pump": 2
    }

    priority_table = {
        "Loft insulation at ceiling level": 1,
        "Low energy lights": 1,
        "Draught proofing of windows and doors": 1,
        "Heating controls for wet central heating": 1,
        "Cavity wall insulation on its own": 2,
        "Double glazed windows": 3,
        "Flat roof insulation": 4,
        "Roof room insulation": 4,
        "Solid wall insulation (external)": 4,
        "Solid wall insulation (internal)": 4,
        "External Wall insulation on system build & Timber frame walls": 4,
        "Air or ground source heat pump": 5
    }

    fabric_priority_table = {
        "Loft insulation at ceiling level": 1,
        "Low energy lights": 2,
        "Draught proofing of windows and doors": 1,
        "Heating controls for wet central heating": 3,
        "Cavity wall insulation on its own": 1,
        "Double glazed windows": 1,
        "Flat roof insulation": 1,
        "Roof room insulation": 1,
        "Solid wall insulation (external)": 1,
        "Solid wall insulation (internal)": 1,
        "External Wall insulation on system build & Timber frame walls": 1,
        "Air or ground source heat pump": 3
    }

    client_priority = {
        "Loft insulation at ceiling level": 1,
        "Low energy lights": 1,
        "Draught proofing of windows and doors": 1,
        "Heating controls for wet central heating": 1,
        "Cavity wall insulation on its own": 1,
        "Double glazed windows": 1,
        "Flat roof insulation": 1,
        "Roof room insulation": 1,
        "Solid wall insulation (external)": 1,
        "Solid wall insulation (internal)": 1,
        "External Wall insulation on system build & Timber frame walls": 1,
        "Air or ground source heat pump": 1
    }

    # Add a new column to calculate total score based on recommendations
    def calculate_recommendation(row, measure_scores, priority_table):
        current_efficiency = row['CURRENT_ENERGY_EFFICIENCY']
        if current_efficiency >= 70:
            return "Already 70"

        total_score = current_efficiency
        selected_measures = []
        recommendations = row['RECOMMENDATION'].split(', ')

        prioritized_measures = sorted(recommendations, key=lambda x: priority_table.get(x.strip(), float('inf')))
        for measure in prioritized_measures:
            score = measure_scores.get(measure.strip(), 0)
            if total_score < 70 and score > 0:
                total_score += score
                selected_measures.append(measure)
            if total_score >= 70:
                break

        return ', '.join(selected_measures) if total_score >= 70 else "Not enough measures"


    # Function to calculate CLIENTS_RECOMMENDATION
    def calculate_client_recommendation(row, measure_scores, client_priority, target_score):
        current_efficiency = row['CURRENT_ENERGY_EFFICIENCY']
        if current_efficiency >= target_score:
            return "Already at target"

        total_score = current_efficiency
        selected_measures = []
        recommendations = row['RECOMMENDATION'].split(', ')

        # Select measures in the order they appear in recommendations (or priority order)
        prioritized_measures = sorted(recommendations, key=lambda x: client_priority.get(x.strip(), float('inf')))
        for measure in prioritized_measures:
            score = measure_scores.get(measure.strip(), 0)
            if total_score < target_score and score > 0:
                total_score += score
                selected_measures.append(measure)
            if total_score >= target_score:
                break

        return ', '.join(selected_measures) if total_score >= target_score else "Not enough measures"

    # Target score provided by the client (replace this with the actual target)
    target_score = target_score

    df['CURRENT_ENERGY_EFFICIENCY'] = pd.to_numeric(df['CURRENT_ENERGY_EFFICIENCY'], errors='coerce').fillna(0)
    df['FINISHING_SAP_SCORE'] = df['RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )

    df['FINISHING_SAP_SCORE'] = pd.to_numeric(df['FINISHING_SAP_SCORE'], errors='coerce').fillna(0)
    df['FINISHING_SAP_SCORE'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['LOWCOST_RECOMMENDATION'] = df.apply(lambda row: calculate_recommendation(row, measure_scores, priority_table), axis=1)
    df['TOTAL_LOWCOST_RECOMMENDATION'] = df['LOWCOST_RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )
    df['TOTAL_LOWCOST_RECOMMENDATION'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['FABRIC_RECOMMENDATION'] = df.apply(lambda row: calculate_recommendation(row, measure_scores, fabric_priority_table), axis=1)
    df['TOTAL_FABRIC_RECOMMENDATION'] = df['FABRIC_RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )
    df['TOTAL_FABRIC_RECOMMENDATION'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['CLIENTS_RECOMMENDATION'] = df.apply(
        lambda row: calculate_client_recommendation(row, measure_scores, client_priority, target_score), axis=1
    )
    df['CLIENT_TARGET_SCORE'] = df['CLIENTS_RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )
    df['CLIENT_TARGET_SCORE'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['CLIENT_TARGET_SCORE'] = df.apply(
        lambda x: 'Not enough measures' if x['CLIENTS_RECOMMENDATION'] == 'Not enough measures' else (
            'same sap score' if x['CLIENTS_RECOMMENDATION'] == '' else x['CLIENT_TARGET_SCORE']
        ), axis=1
    )

    cols_to_replace = ['TOTAL_LOWCOST_RECOMMENDATION', 'TOTAL_FABRIC_RECOMMENDATION', 'CLIENT_TARGET_SCORE']

    for col in cols_to_replace:
        df[col] = df.apply(
            lambda x: 'same sap score' if x[col] == x['CURRENT_ENERGY_EFFICIENCY'] else x[col], axis=1
        )


    df1 = pd.read_excel(r"data/ECO4 Full Project Scores Matrix.xlsx")
    # SAP rating band ranges
    sap_band_ranges = [
        (0, 10.4, "Low_G"),
        (10.5, 20.4, "High_G"),
        (20.5, 29.4, "Low_F"),
        (29.5, 38.4, "High_F"),
        (38.5, 46.4, "Low_E"),
        (46.5, 54.4, "High_E"),
        (54.5, 61.4, "Low_D"),
        (61.5, 68.4, "High_D"),
        (68.5, 74.4, "Low_C"),
        (74.5, 80.4, "High_C"),
        (80.5, 85.9, "Low_B"),
        (86.0, 91.4, "High_B"),
        (91.5, 95.9, "Low_A"),
        (96.0, float('inf'), "High_A")
    ]

    # Assign SAP band based on ranges
    def assign_sap_band(value):
        for lower, upper, band in sap_band_ranges:
            if lower <= value <= upper:
                return band
        return None

    # Dynamically determine Floor Area Segments from df1
    def parse_floor_area_segments(segment):
        try:
            if "+" in segment:  # Handle open-ended ranges like "200+"
                lower = int(segment.replace("+", "").strip())
                upper = float('inf')
            else:
                lower, upper = map(int, segment.split("-"))
            return (lower, upper, segment)
        except ValueError as e:
            print(f"Error parsing segment: {segment}")
            return None

    ## Ensure TOTAL_FLOOR_AREA is numeric
    df['TOTAL_FLOOR_AREA'] = pd.to_numeric(df['TOTAL_FLOOR_AREA'], errors='coerce').fillna(0)

    # Parse Floor Area Segments from df1
    floor_area_segments = [
        parse_floor_area_segments(seg)
        for seg in df1["Floor Area Segment"].unique()
        if parse_floor_area_segments(seg) is not None
    ]

    # Function to assign floor area segment
    def assign_floor_area_segment(value):
        for lower, upper, segment in floor_area_segments:
            if lower <= value <= upper:
                return segment
        return None

    # Updated get_cost_savings function
    def get_cost_savings(row):
        # Step 1: Get SAP bands for CURRENT_ENERGY_EFFICIENCY and FINISHING_SAP_SCORE
        current_band = assign_sap_band(row["CURRENT_ENERGY_EFFICIENCY"])
        finishing_band = assign_sap_band(row["FINISHING_SAP_SCORE"])
        
        # Debugging outputs
        #print(f"Row: {row.name}, Current Band: {current_band}, Finishing Band: {finishing_band}")

        # Step 2: Get Floor Area Segment
        floor_area_segment = assign_floor_area_segment(row["TOTAL_FLOOR_AREA"])
        #print(f"Row: {row.name}, Floor Area Segment: {floor_area_segment}")

        # Step 3: Match with df1 to find Cost Savings
        filtered_df1 = df1[
            (df1["Floor Area Segment"] == floor_area_segment) &
            (df1["Starting Band"] == current_band) &
            (df1["Finishing Band"] == finishing_band)
        ]

        # Debugging outputs
        #print(f"Row: {row.name}, Filtered df1:\n{filtered_df1}")

        if not filtered_df1.empty:
            return filtered_df1["Cost Savings"].iloc[0]
        else:
            return "Not Found"

    # Apply get_cost_savings
    df["COST_SAVINGS"] = df.apply(get_cost_savings, axis=1)
    df["COST_SAVINGS_RECOM"] = df.apply(lambda row: f"{assign_sap_band(row['CURRENT_ENERGY_EFFICIENCY'])} -> {assign_sap_band(row['FINISHING_SAP_SCORE'])}", axis=1)

    df2 = pd.read_excel(r"data/coststempo.xlsx")

    # Mapping dictionary
    recommendation_mapping = {
        "Air or ground source heat pump": "ASHP",
        "Double glazed windows": "DGW",
        "Draught proofing of windows and doors": "DPW",
        "Cavity wall insulation on its own": "CWI",
        "Flat roof insulation": "FRI",
        "Heating controls for wet central heating": "HC",
        "Loft insulation at ceiling level": "LI",
        "Roof room insulation": "RI",
        "Low energy lights": "LEL",
        "Solid wall insulation (external)": "SWEI",
        "Solid wall insulation (internal)": "SWII",
        "External Wall insulation on system build & Timber frame walls": "EWI"
    }

    # Step 1: Combine `PROPERTY_TYPE` and `BUILT_FORM` to create `Property Type : Build Form`
    df["Property Type : Build Form"] = df["PROPERTY_TYPE"] + " : " + df["BUILT_FORM"]

    # Step 2: Define function to calculate recommendation sums
    def calculate_recommendation_sum(row, recommendation_column, df, mapping, subset=None):
        recommendations = row[recommendation_column].split(", ") if row[recommendation_column] else []
        if subset:
            recommendations = [rec for rec in recommendations if rec in subset]

        
        columns_to_sum = [mapping[rec] for rec in recommendations if rec in mapping]
        
        if not columns_to_sum:  # If no valid recommendations are found
            return row.get(recommendation_column, 0)  # Return existing column value
        
        matching_row = df[df["Property Type : Build Form"] == row["Property Type : Build Form"]]
        if not matching_row.empty:
            return matching_row[columns_to_sum].sum(axis=1).values[0]
        
        return row.get(recommendation_column, 0)  # Default to existing column value if no match is found

    # Step 3: Compute recommendation sums
    df["RECOMMENDATION_SUM"] = df.apply(
        calculate_recommendation_sum, axis=1, args=("RECOMMENDATION", df2, recommendation_mapping)
    )
    df["LOWCOST_RECOMMENDATION_SUM"] = df.apply(
        calculate_recommendation_sum, axis=1, args=("LOWCOST_RECOMMENDATION", df2, recommendation_mapping)
    )
    df["FABRIC_RECOMMENDATION_SUM"] = df.apply(
        calculate_recommendation_sum, axis=1, args=("FABRIC_RECOMMENDATION", df2, recommendation_mapping)
    )

    df = df[df['RECOMMENDATION'] != ""]

    return df





# Predefined credentials
VALID_EMAIL = "banavasipavan2002@gmail.com"
VALID_PASSWORD = "Pavan$2002"

def login():
    st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            font-family: Arial, sans-serif;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Login Page")

    # Input fields for credentials
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    # Handle login
    if login_button:
        if email == VALID_EMAIL and password == VALID_PASSWORD:
            # Set login state
            st.session_state.logged_in = True
            
            # Use query parameters to reload the page
            st.query_params = {"logged_in": True}  # Simulate a page reload by updating query params
            
            # Show success message
            st.success("Login successful!, Press Login again")
        else:
            st.error("Invalid email or password.")




def main_app():
    st.title("Recommendation Generator")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

    # Dropdown for recommendations
    recommendations_list = [
        "Air or ground source heat pump",
        "Double glazed windows",
        "Draught proofing of windows and doors",
        "Cavity wall insulation on its own",
        "Flat roof insulation",
        "Heating controls for wet central heating",
        "Loft insulation at ceiling level",
        "Low energy lights",
        "Roof room insulation",
        "Solid wall insulation (external)",
        "Solid wall insulation (internal)",
        "External Wall insulation on system build & Timber frame walls"
    ]
    selected_recommendations = st.multiselect("Select Recommendations", recommendations_list)

    # Target score slider
    target_score = st.slider("Set Target Score", min_value=0, max_value=100, value=70, step=1)

    # Display selected recommendations as a cart
    if selected_recommendations:
        st.write("### Selected Recommendations:")
        for rec in selected_recommendations:
            st.write(f"- {rec}")

    # Placeholder for file processing logic
    if uploaded_file:
        st.write("Processing file...")
        # Call your file processing function here
        processed_df = process_file(uploaded_file, selected_recommendations, target_score)
        if processed_df is not None:
            st.write("### Processed Data")
            st.dataframe(processed_df)

            # Download processed data
            processed_file_name = "processed_recommendations.xlsx"
            st.write(processed_df.shape[0])
            processed_df.to_excel(processed_file_name, index=False)
            st.download_button(
                label="Download Processed File",
                data=open(processed_file_name, "rb"),
                file_name=processed_file_name,
                mime="application/vnd.ms-excel"
            )
        # Placeholder logic
        st.write("File processed successfully!")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.query_params.get("logged_in") == "True" or st.session_state.logged_in:
    main_app()  # Display the main app if logged in
else:
    login()  # Display the login page
