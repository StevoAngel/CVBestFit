import json
import ollama
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

#Cargar modelo de Ollama:
try:
    ollama.pull('mistral')
    llm = ChatOllama(model="mistral")
    print("Modelo de Ollama cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo de Ollama: {e}")
    llm = None

# Definir estructura de datos para la informacion extraida del CV:
class CVInfo(BaseModel):
    name: str = Field(description="Nombre del candidato")
    email: str = Field(description="Email del candidato")
    phone: str = Field(description="Telefono del candidato")
    profile: str = Field(description="Resumen del perfil del candidato")
    location: str = Field(description="Ubicacion del candidato")
    experience: list[str] = Field(description="Experiencia laboral del candidato")
    education: list[str] = Field(description="Educacion del candidato")
    hardSkills: list[str] = Field(description="Habilidades del candidato")
    softSkills: list[str] = Field(description="Habilidades blandas del candidato")

class jobDescriptionInfo(BaseModel):
    title: str = Field(description="Titulo del trabajo")
    description: str = Field(description="Descripcion del trabajo")
    responsibilities: list[str] = Field(description="Responsabilidades del trabajo")
    hardSkills: list[str] = Field(description="Habilidades duras requeridas")
    softSkills: list[str] = Field(description="Habilidades blandas requeridas")
    location: str = Field(description="Ubicacion del trabajo")


class LLMParser:
    def extract_CVInfo(self, cvText, verbose=False):
        """Extrae la información del CV utilizando un modelo de lenguaje."""
        self.cvText = cvText

        # Se define el prompt para extraer la informacion del CV:
        prompt = f"""
        Eres un experto en extracción de información de currículums en inglés y español. Tu tarea es generar **únicamente** un JSON **válido y correctamente formateado** siguiendo el esquema siguiente:
        {{
            "name": "string",
            "email": "string",
            "phone": "string",
            "location": "string",
            "profile": "string",
            "experience": ["string", "string"],
            "education": ["string", "string"],
            "hardSkills": ["string", "string"],
            "softSkills": ["string", "string"],
        }}

        Reglas:
        - Las respuestas deben ser en inglés o español, según el idioma del texto de entrada.
        - "profile" debe ser un extracto **representativo** en **150 palabras** del currículum enfocado especialmente en experiencia, habilidades y educación del candidato en str **texto plano** que será utilizada para crear una similitud vectorial con la descripción de la vacante.
        - Los campos de softSkills y hardSkills deben estar correctamente diferenciados, no debe haber una habilidad dura en el campo de habilidades blandas ni viceversa.
        - **Debes incluir única y exclusivamente los campos mencionados en el esquema y en ese formato.
        - El resultado debe ser solo un **JSON puro**, sin explicaciones, texto adicional, ni etiquetas HTML, ni signos de puntuación extra.
        - **Todos los textos (strings) deben ir entre comillas dobles**.
        - Si un campo no está presente en el CV, déjalo vacío o usa una lista vacía (`[]`) según corresponda.
        - Sí encuentras más de un valor para un campo tipo string, debes incluir solo el primero.
        - **No incluyas comas sobrantes al final de listas o diccionarios**.
        - Al final valida que estén todos los campos requeridos según el esquema y que estén correctamente formateados.

        Ahora extrae la información del siguiente currículum y devuelve el JSON válido:
        {self.cvText}
        """
        # Se llama al modelo de lenguaje para extraer la información:
        response = llm.invoke([
            {"role": "user", "content": prompt}
        ])

        if verbose:
            print(f"CV Information Response:\n{response.content}")
    
        # Se convierte la respuesta en un objeto JSON:
        rawInfo = json.loads(response.content)
        cvInfoJSON = CVInfo(**rawInfo)

        return cvInfoJSON
    
    def extract_jobDescriptionInfo(self, jobDescriptionText, verbose=False):
        """Extrae la información del puesto de trabajo utilizando un modelo de lenguaje."""
        self.jobDescriptionText = jobDescriptionText

        #Se define el prompt para extraer la informacion del puesto de trabajo:
        prompt = f"""
        Eres un experto en extracción de descripciones de puestos de trabajo en inglés y español. Tu tarea es generar **únicamente** un JSON **válido y correctamente formateado** siguiendo el esquema siguiente:
        {{
            "title": "string",
            "description": "string",
            "responsibilities": ["string", "string"],
            "hardSkills": ["string", "string"],
            "softSkills": ["string", "string"],
            "location": "string"
        }}

        Reglas:
        - Las respuestas deben ser en inglés o español, según el idioma del texto de entrada.
        - Los campos de softSkills y hardSkills deben estar correctamente diferenciados, no debe haber una habilidad dura en el campo de habilidades blandas ni viceversa.
        - **Debes incluir única y exclusivamente los campos mencionados en el esquema y en ese formato.
        - El resultado debe ser solo un **JSON puro**, sin explicaciones, texto adicional, ni etiquetas HTML, ni signos de puntuación extra.
        - **Todos los textos (strings) deben ir entre comillas dobles**.
        - Si un campo no está presente en el CV, déjalo vacío o usa una lista vacía (`[]`) según corresponda.
        - Sí encuentras más de un valor para un campo tipo string, debes incluir solo el primero.
        - **No incluyas comas sobrantes al final de listas o diccionarios**.
        - Al final valida que estén todos los campos requeridos según el esquema y que estén correctamente formateados.

        Ahora extrae la información del siguiente currículum y devuelve el JSON válido:
        {self.jobDescriptionText}
        """
        # Se llama al modelo de lenguaje para extraer la información:
        response = llm.invoke([
            {"role": "user", "content": prompt}
        ])
        if verbose:
            print(f"Job Description Information Response:\n{response.content}")
        # Se convierte la respuesta en un objeto JSON:
        rawInfo = json.loads(response.content)
        jobDescriptionInfoJSON = jobDescriptionInfo(**rawInfo)

        return jobDescriptionInfoJSON
