import streamlit as st
import pandas as pd
import re


# Age band dictionary
age_band_dict = {
    "A": "England and Wales: before 1900", "B": "England and Wales: 1900-1929", "C": "England and Wales: 1930-1949", "D": "England and Wales: 1950-1966",
    "E": "England and Wales: 1967-1975", "F": "England and Wales: 1976-1982", "G": "England and Wales: 1983-1990", "H": "England and Wales: 1991-1995",
    "I": "England and Wales: 1996-2002", "J": "England and Wales: 2003-2006", "K": "England and Wales: 2007-2011", "L": "England and Wales: 2012-2022", "M": "England and Wales: 2023 onwards"
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
    "Solid wall insulation (internal or external)",
    "External Wall insulation on system build & Timber frame walls",
    "Solar PV",
    "Insulated Doors",
    "Hotwater cylinder Insulation"
]

# Function to append recommendations
def append_recommendation(current, new):
    return f"{current}, {new}" if current else new

def process_file(file, selected_recommendations, target_score):
    import chardet  # For detecting file encoding

    # Try reading the file without specifying encoding first
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, dtype=str)  # Read all columns as strings (default encoding)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file, dtype=str)  # Read all columns as strings
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        # If reading fails, try to handle encoding errors
        st.warning("An error occurred while reading the file. Trying to detect encoding...")
        
        # Detect encoding using chardet
        raw_data = file.read()
        file.seek(0)  # Reset file pointer after reading
        detected_encoding = chardet.detect(raw_data)["encoding"]
        
        # Try reading the file again with the detected encoding
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, dtype=str, encoding=detected_encoding)  # Use detected encoding
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, dtype=str)  # Excel doesn't need encoding (handled by default)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return None
        except Exception as encoding_error:
            st.error(f"Failed to read the file with detected encoding '{detected_encoding}'. Error: {encoding_error}")
            return None

    # Ensure the file contains data
    if df.empty:
        st.error("The uploaded file is empty.")
        return None
    
    df['CONSTRUCTION_AGE_BAND'] = df['CONSTRUCTION_AGE_BAND'].fillna("").astype(str)
    # Standardize BUILT_FORM field
    df['BUILT_FORM'] = df['BUILT_FORM'].str.replace('Enclosed ', '', regex=False)
    df['MULTI_GLAZE_PROPORTION'] = df['MULTI_GLAZE_PROPORTION'].replace(r'\xa0', '', regex=True).str.strip()

    # Initialize the RECOMMENDATION column
    df['RECOMMENDATION'] = ""

    # Apply selected recommendation logic hereS
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

        if recommendation == "Solid wall insulation (internal or external)":
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

        if recommendation == "Solar PV":
            df['PHOTO_SUPPLY'] = pd.to_numeric(df['PHOTO_SUPPLY'], errors='coerce').fillna(0).astype('int64')
            df['PHOTO_SUPPLY'] = df['PHOTO_SUPPLY'].replace(0.0, 0)
            condition = (
                (df['PHOTO_SUPPLY'] == 0.0) |  (df['PHOTO_SUPPLY'] == 0)
            )
            df.loc[condition, 'RECOMMENDATION'] = df.loc[condition, 'RECOMMENDATION'].apply(
                lambda rec: append_recommendation(rec, recommendation)
            )

        if recommendation == "Insulated Doors":
            df['MULTI_GLAZE_PROPORTION'] = pd.to_numeric(df['MULTI_GLAZE_PROPORTION'], errors='coerce').fillna(0).astype('int64')
            condition = (df['GLAZED_TYPE'].str.lower() == 'single glazing')
            df.loc[condition, 'RECOMMENDATION'] = df.loc[condition, 'RECOMMENDATION'].apply(
                lambda rec: append_recommendation(rec, recommendation)
            )
            
        if recommendation == "Hotwater cylinder Insulation":
            df['HOTWATER_DESCRIPTION'] = df['HOTWATER_DESCRIPTION'].fillna('').str.lower()
            df['HOT_WATER_ENERGY_EFF'] = df['HOT_WATER_ENERGY_EFF'].fillna('').str.lower()

            # Define the condition using a regex pattern with OR operator (|)
            condition = (
                df['HOTWATER_DESCRIPTION'].str.contains(r'from main system, no cylinder thermostat', regex=True) &
                df['HOT_WATER_ENERGY_EFF'].str.contains(r'average|poor|very poor', regex=True)
            )

            df.loc[condition, 'RECOMMENDATION'] = df.loc[condition, 'RECOMMENDATION'].apply(
                lambda rec: append_recommendation(rec, recommendation)
            )


    # Clean up trailing commas in RECOMMENDATION column
    df['RECOMMENDATION'] = df['RECOMMENDATION'].str.rstrip(', ')
    
    # Scores and priority tables
    measure_scores = {
        "Loft insulation at ceiling level": 2,
        "Low energy lights": 1,
        "Draught proofing of windows and doors": 1,
        "Heating controls for wet central heating": 4,
        "Cavity wall insulation on its own": 6,
        "Double glazed windows": 2,
        "Flat roof insulation": 3,
        "Roof room insulation": 5,
        "Solid wall insulation (internal or external)": 5,
        "External Wall insulation on system build & Timber frame walls": 5,
        "Air or ground source heat pump": 2,
        "Solar PV": 10,
        "Insulated Doors": 1,
        "Hotwater cylinder Insulation": 2
    }

    priority_table = {
        "Solar PV" : 1,
        "Loft insulation at ceiling level": 1,
        "Low energy lights": 1,
        "Draught proofing of windows and doors": 1,
        "Heating controls for wet central heating": 1,
        "Insulated Doors": 1,
        "Hotwater cylinder Insulation": 1,
        "Cavity wall insulation on its own": 2,
        "Double glazed windows": 3,
        "Flat roof insulation": 4,
        "Roof room insulation": 4,
        "Solid wall insulation (internal or external)": 4,
        "External Wall insulation on system build & Timber frame walls": 4,
        "Air or ground source heat pump": 5,

    }

    fabric_priority_table = {
        "Solar PV" : 1,
        "Loft insulation at ceiling level": 1,
        "Low energy lights": 2,
        "Draught proofing of windows and doors": 1,
        "Insulated Doors": 1,
        "Hotwater cylinder Insulation": 1,
        "Heating controls for wet central heating": 3,
        "Cavity wall insulation on its own": 1,
        "Double glazed windows": 1,
        "Flat roof insulation": 1,
        "Roof room insulation": 1,
        "Solid wall insulation (internal or external)": 1,
        "External Wall insulation on system build & Timber frame walls": 1,
        "Air or ground source heat pump": 3
    }

    client_priority = {
        "Solar PV" : 1,
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
        "Air or ground source heat pump": 1,
        "Insulated Doors": 1,
        "Hotwater cylinder Insulation": 1
    }

    # Add a new column to calculate total score based on recommendations
    def calculate_recommendation(row, measure_scores, priority_table):
        current_efficiency = row['CURRENT_ENERGY_EFFICIENCY']
        if current_efficiency >= 70:
            return "SAP score already meets 70"

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

        return ', '.join(selected_measures) if total_score >= 70 else "Property cannot reach 70"


    # Function to calculate CLIENTS_RECOMMENDATION
    def calculate_client_recommendation(row, measure_scores, client_priority, target_score):
        current_efficiency = row['CURRENT_ENERGY_EFFICIENCY']
        if current_efficiency >= target_score:
            return "Client Target SAP Score Achieved"

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

        return ', '.join(selected_measures) if total_score >= target_score else "Property cannot reach client target SAP score"

    # Target score provided by the client (replace this with the actual target)
    target_score = target_score
    
    df['Renewable_energy'] = df['RECOMMENDATION'].apply(
        lambda rec: ', '.join([r for r in rec.split(', ') if r.strip() in ['Solar PV', 'Air or ground source heat pump']]) 
        if any(r.strip() in ['Solar PV', 'Air or ground source heat pump'] for r in rec.split(', ')) else 'Not Found'
    )

    df['CURRENT_ENERGY_EFFICIENCY'] = pd.to_numeric(df['CURRENT_ENERGY_EFFICIENCY'], errors='coerce').fillna(0)
    df['FINISHING_SAP_SCORE'] = df['RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )

    df['FINISHING_SAP_SCORE'] = pd.to_numeric(df['FINISHING_SAP_SCORE'], errors='coerce').fillna(0)
    df['FINISHING_SAP_SCORE'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['LOWCOST_EPC_C_RECOMMENDATION'] = df.apply(lambda row: calculate_recommendation(row, measure_scores, priority_table), axis=1)
    df['TOTAL_LOWCOST_EPC_C_RECOMMENDATION'] = df['LOWCOST_EPC_C_RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )
    df['TOTAL_LOWCOST_EPC_C_RECOMMENDATION'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['FABRIC_FIRST_EPC_C_RECOMMENDATION'] = df.apply(lambda row: calculate_recommendation(row, measure_scores, fabric_priority_table), axis=1)
    df['TOTAL_FABRIC_FIRST_EPC_C_RECOMMENDATION'] = df['FABRIC_FIRST_EPC_C_RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )
    df['TOTAL_FABRIC_FIRST_EPC_C_RECOMMENDATION'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['CLIENTS_RECOMMENDATION'] = df.apply(
        lambda row: calculate_client_recommendation(row, measure_scores, client_priority, target_score), axis=1
    )
    df['CLIENT_TARGET_SCORE'] = df['CLIENTS_RECOMMENDATION'].apply(
        lambda rec: sum(measure_scores.get(r.strip(), 0) for r in rec.split(', ') if r.strip())
    )
    df['CLIENT_TARGET_SCORE'] += df['CURRENT_ENERGY_EFFICIENCY']

    df['CLIENT_TARGET_SCORE'] = df.apply(
        lambda x: 'Property cannot reach client target SAP score' if x['CLIENTS_RECOMMENDATION'] == 'Property cannot reach client target SAP score' else (
            'same sap score' if x['CLIENTS_RECOMMENDATION'] == '' else x['CLIENT_TARGET_SCORE']
        ), axis=1
    )

    cols_to_replace = ['TOTAL_LOWCOST_EPC_C_RECOMMENDATION', 'TOTAL_FABRIC_FIRST_EPC_C_RECOMMENDATION', 'CLIENT_TARGET_SCORE']

    for col in cols_to_replace:
        df[col] = df.apply(
            lambda x: 'SAP Score Unchanged' if x[col] == x['CURRENT_ENERGY_EFFICIENCY'] else x[col], axis=1
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
            return "Nill"

    # Apply get_cost_savings
    df["COST_SAVINGS"] = df.apply(get_cost_savings, axis=1)
    #df["COST_SAVINGS_RECOM"] = df.apply(lambda row: f"{assign_sap_band(row['CURRENT_ENERGY_EFFICIENCY'])} -> {assign_sap_band(row['FINISHING_SAP_SCORE'])}", axis=1)
                        
    return df



# Predefined credentials
VALID_EMAIL = "designspecifics@dev"
VALID_PASSWORD = "DSL@2025"

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


def get_energy_rating(score):
    """Maps the energy efficiency score to an energy rating (A-G)."""
    if score >= 92:
        return "A"
    elif score >= 81:
        return "B"
    elif score >= 69:
        return "C"
    elif score >= 55:
        return "D"
    elif score >= 39:
        return "E"
    elif score >= 21:
        return "F"
    else:
        return "G"

def custom_slider(label="Set Target Score", default_value=70, min_value=0, max_value=100, step=1):
    """Creates a slider with proper alignment and spacing."""
    
    # Streamlit slider to select a value
    slider_value = st.slider(label, min_value, max_value, default_value, step)

    # Get the corresponding energy rating
    rating = get_energy_rating(slider_value)

    # Add custom CSS for styling
    st.markdown(
    """
    <style>
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 16px;
        }
        .slider {
            width: 300px;
            margin-right: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="slider-container">
            <span style="color: red;">{slider_value}</span>
            <span style="color: red;">{rating}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return slider_value  # Return the selected score


def main_app():
    st.title("Recommendation Generator")

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Read the uploaded file and extract column names
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, dtype=str)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, dtype=str)

        column_list = list(df.columns)

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
            "Solid wall insulation (internal or external)",
            "External Wall insulation on system build & Timber frame walls",
            "Solar PV",
            "Insulated Doors",
            "Hotwater cylinder Insulation"
        ]

        selected_recommendations = st.multiselect("Select Recommendations", recommendations_list)

        st.write("### Select Analysis Type")
        lowcost_epc = st.checkbox("Lowcost EPC C")
        fabric_cost_epc = st.checkbox("Fabric First EPC C")
        full_recommendations = st.checkbox("All Possible Recommendations")
        include_renewable_energy = st.checkbox("Renewable Energy")
        cost_savings = st.checkbox("Cost Savings")
        client_target_epc = st.checkbox("Client Target SAP Score")

        target_score = 70  # Default target score

        additional_columns = []

        if lowcost_epc:
            additional_columns.extend(["LOWCOST_EPC_C_RECOMMENDATION", "TOTAL_LOWCOST_EPC_C_RECOMMENDATION"])

        if fabric_cost_epc:
            additional_columns.extend(["FABRIC_FIRST_EPC_C_RECOMMENDATION", "TOTAL_FABRIC_FIRST_EPC_C_RECOMMENDATION"])

        if full_recommendations:
            additional_columns.extend(["RECOMMENDATION", "FINISHING_SAP_SCORE"])

        if include_renewable_energy:
            additional_columns.append("Renewable_energy")

        if client_target_epc:
            target_score = custom_slider("Set Target Score", default_value=70, min_value=0, max_value=100, step=1)
            additional_columns.extend(["CLIENTS_RECOMMENDATION", "CLIENT_TARGET_SCORE"])

        if cost_savings:
            additional_columns.append("COST_SAVINGS")
        
         # Extend column_list with additional columns
        column_list.extend([col for col in additional_columns if col not in column_list])

        # Add vertical spacing above the button
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

        if st.button("Generate Recommendations"):
            with st.spinner("Processing data..."):
                processed_df = process_file(uploaded_file, selected_recommendations, target_score)
                if processed_df is not None:
                    processed_df = processed_df[column_list]
                    st.write("### Processed Data")
                    st.dataframe(processed_df)
                    st.write(processed_df.shape[0])
                    processed_file_name = "processed_recommendations.xlsx"
                    processed_df.to_excel(processed_file_name, index=False)

                    st.download_button(
                        label="Download Processed File",
                        data=open(processed_file_name, "rb"),
                        file_name=processed_file_name,
                        mime="application/vnd.ms-excel"
                    )

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.query_params.get("logged_in") == "True" or st.session_state.logged_in:
    main_app()  # Display the main app if logged in
else:
    login()  # Display the login page
