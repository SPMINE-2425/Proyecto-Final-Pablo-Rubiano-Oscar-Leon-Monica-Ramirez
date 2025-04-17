import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Abandono Escolar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎓 Sistema de Predicción de Riesgo de Abandono Escolar")

# Sidebar
st.sidebar.header("Información del Estudiante")

# Función para cargar los datos del modelo
@st.cache_resource
def load_model_info():
    try:
        response = requests.get("http://localhost:8000/model_info")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Formulario de entrada
with st.sidebar.form("student_form"):
    st.subheader("Datos Académicos")
    application_mode = st.selectbox(
        "Modo de Aplicación",
        options=list(range(1, 18)),
        help="Método por el cual el estudiante se inscribió"
    )
    application_order = st.number_input(
        "Orden de Aplicación",
        min_value=0,
        max_value=10,
        value=1
    )
    previous_qualification = st.selectbox(
        "Calificación Previa",
        options=list(range(1, 20)),
        index=9
    )
    prev_qualification_grade = st.slider(
        "Nota de Calificación Previa",
        0.0, 200.0, 120.0
    )
    admission_grade = st.slider(
        "Nota de Admisión",
        0.0, 200.0, 120.0
    )
    
    st.subheader("Datos Personales")
    marital_status = st.selectbox(
        "Estado Civil",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Soltero", "Casado", "Viudo", "Divorciado", "Separado", "Unión civil"][x-1]
    )
    daytime_attendance = st.radio(
        "Asistencia",
        options=[1, 0],
        format_func=lambda x: "Diurno" if x == 1 else "Nocturno"
    )
    gender = st.radio(
        "Género",
        options=[1, 0],
        format_func=lambda x: "Masculino" if x == 1 else "Femenino"
    )
    age = st.number_input(
        "Edad al Matricularse",
        min_value=15, max_value=70, value=20
    )
    
    st.subheader("Situación Económica")
    debtor = st.checkbox("¿Es deudor?")
    tuition_fees = st.checkbox("¿Matrícula al día?", value=True)
    scholarship = st.checkbox("¿Becado?")
    
    submitted = st.form_submit_button("Predecir Riesgo")

# Resultados
if submitted:
    student_data = {
        "marital_status": marital_status,
        "application_mode": application_mode,
        "application_order": application_order,
        "daytime_evening_attendance": daytime_attendance,
        "previous_qualification": previous_qualification,
        "previous_qualification_grade": prev_qualification_grade,
        "admission_grade": admission_grade,
        "debtor": debtor,
        "tuition_fees_up_to_date": tuition_fees,
        "gender": gender,
        "scholarship_holder": scholarship,
        "age_at_enrollment": age
    }
    
    try:
        with st.spinner("Analizando riesgo..."):
            response = requests.post("http://localhost:8000/predict", json=student_data)
            
        if response.status_code == 200:
            result = response.json()
            
            # Mostrar resultados
            st.success("Análisis completado")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Predicción",
                    value="Alto Riesgo" if result['prediction'] == 0 else "Bajo Riesgo",
                    delta=f"{result['probability']*100:.1f}% de probabilidad"
                )
                
            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    [result['probability'], 1-result['probability']],
                    labels=['Riesgo', 'Seguro'],
                    colors=['#ff4b4b', '#2ecc71'],
                    autopct='%1.1f%%'
                )
                st.pyplot(fig)
            
            # Recomendaciones
            st.subheader("Recomendaciones")
            if result['prediction'] == 0:
                st.warning("""
                🔍 Este estudiante muestra alto riesgo de abandono. Recomendamos:
                - Contactar al estudiante para evaluación personalizada
                - Ofrecer tutoría académica
                - Revisar situación económica
                """)
            else:
                st.info("""
                ✅ Este estudiante tiene bajo riesgo de abandono. Recomendamos:
                - Monitoreo periódico
                - Ofrecer oportunidades de enriquecimiento académico
                """)
                
        else:
            st.error(f"Error en la predicción: {response.text}")
            
    except Exception as e:
        st.error(f"No se pudo conectar al servidor: {str(e)}")
        st.info("Asegúrate que el servidor de la API está corriendo en http://localhost:8000")

# Información del modelo
model_info = load_model_info()
if model_info:
    with st.expander("ℹ️ Información Técnica del Modelo"):
        st.write("**Características utilizadas:**", model_info['feature_columns'])
        st.write("**Clases objetivo:**", model_info.get('target_classes', ['Dropout', 'Graduate']))
else:
    st.warning("No se pudo cargar la información del modelo. La API puede no estar disponible.")

# Footer
st.markdown("---")
st.markdown("Sistema desarrollado para el proyecto final de Seminario de Programación")