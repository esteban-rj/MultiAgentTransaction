import os
import glob
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from pymongo import MongoClient
# from langchain_ollama.llms import OllamaLLM # Switch to OpenAI
# from langchain_community.embeddings import OllamaEmbeddings # Switch to OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # For Docker Model Runner
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore # Simplified for example
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongodb:27017/")
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "cocina_db")
# Updated for Docker Model Runner (OpenAI Compatible)
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview") # For Chat LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Must be set in environment
OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-3-small") # Dedicated embeddings model
DOCUMENTS_DIR = "/app/documentos"

# FastAPI App
app = FastAPI()

# MongoDB Client
client = MongoClient(MONGODB_URI)
db = client[MONGO_DATABASE]
ingredients_collection = db["ingredientes_disponibles"]

# LLM Initialization
if not OPENAI_API_KEY:
    print("CRITICAL: OPENAI_API_KEY environment variable not set. The application will not work.")
    # Optionally, raise an error or exit if you want to prevent startup without API key
    # raise ValueError("OPENAI_API_KEY environment variable not set.")

llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_base=OPENAI_API_BASE,
    openai_api_key=OPENAI_API_KEY,
)
embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDINGS_MODEL_NAME, # Use dedicated embeddings model
    openai_api_base=OPENAI_API_BASE,
    openai_api_key=OPENAI_API_KEY,
)

# --- Knowledge Base for Recipes (PDFs) ---
pdf_docs = []
for pdf_path in glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")):
    try:
        loader = PyPDFLoader(pdf_path)
        pdf_docs.extend(loader.load())
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
recipe_texts = text_splitter.split_documents(pdf_docs)

# Simple in-memory vector store for recipe documents
# For production, consider FAISS, Chroma, etc.
if recipe_texts:
    recipe_vector_store = InMemoryVectorStore.from_documents(recipe_texts, embeddings)
    recipe_retriever = recipe_vector_store.as_retriever()
    recipe_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=recipe_retriever,
        return_source_documents=True
    )
else:
    recipe_qa_chain = None
    print("Warning: No PDF recipes found or loaded. Recipe agent will not have knowledge.")

# --- Agent Tools ---

# Tool 1: Check available ingredients
def check_inventory(query: str) -> str:
    """Checks the MongoDB database for available ingredients and their quantities based on the user's query.
    The query should list the ingredients needed and their quantities.
    This tool will return a list of available ingredients and their current stock, 
    and note which ingredients are missing or insufficient."""
    print(f"Inventory Agent: Received query - {query}")
    # This is a simplified approach. A more robust solution would involve the LLM parsing the recipe 
    # to extract ingredient names and quantities, then querying MongoDB for each.
    # For now, we assume the `query` to this tool might be a general question about ingredients or a specific list.
    
    # Let the LLM try to understand what ingredients are being asked for based on the query.
    # This is a placeholder for a more sophisticated ingredient extraction logic.
    # For a real system, you'd have the LLM parse the recipe text from the RecipeAgent first.
    
    # Example: if query is "Do we have tomatoes and onions for pasta?"
    # A more advanced system would parse this. Here, we'll make a simpler assumption or have the LLM try.

    response_lines = []
    # This is highly dependent on how the supervisor structures the query to this tool.
    # For now, let's assume the query might be "pasta carbonara ingredients list"
    # and the LLM should figure out what to ask the DB.
    # A better way: the supervisor extracts ingredients from recipe agent, then queries inventory.

    # Simplified: let's try to match some keywords from the query to ingredients in DB
    # This is not robust. The LLM should ideally generate specific ingredient queries.
    ai_message = llm.invoke(
        f"Based on the recipe query '{query}', list the key ingredients likely needed, comma-separated. Example: Tomates, Cebolla, Ajo"
    )
    requested_ingredients_str = ai_message.content.strip() # Access .content attribute
    
    if not requested_ingredients_str:
        return "Inventory Agent: Could not determine specific ingredients from the query to check inventory."

    requested_ingredients = [ing.strip() for ing in requested_ingredients_str.split(',') if ing.strip()]
    
    if not requested_ingredients:
        return "Inventory Agent: No specific ingredients identified to check inventory."

    response_lines.append("Estado del inventario para ingredientes consultados:")
    all_found = True
    for ing_name in requested_ingredients:
        # Basic case-insensitive search for flexibility
        ingredient_data = ingredients_collection.find_one({"nombre": {"$regex": f"^{ing_name}$", "$options": "i"}})
        if ingredient_data:
            response_lines.append(f"- {ingredient_data['nombre']}: {ingredient_data['cantidad_gramos']}g disponibles.")
        else:
            response_lines.append(f"- {ing_name}: No encontrado en el inventario.")
            all_found = False
    
    if not all_found:
        response_lines.append("Algunos ingredientes solicitados no están en el inventario.")
    
    return "\n".join(response_lines)

# Tool 2: Get recipe details
def get_recipe_details(query: str) -> str:
    """Retrieves information about a specific recipe from the PDF documents.
    The query should be the name of the recipe or a question about its ingredients/preparation."""
    print(f"Recipe Agent: Received query - {query}")
    if not recipe_qa_chain:
        return "Recipe Agent: Knowledge base for recipes is not available (no PDFs loaded)."
    try:
        result = recipe_qa_chain({"query": query})
        # Focus on the answer, but could include sources if useful
        answer = result.get('result', "No se encontró información para esta receta.")
        # source_docs = result.get('source_documents')
        # if source_docs:
        #     answer += f"\n(Fuentes: {[doc.metadata.get('source', 'N/A') for doc in source_docs]})"
        return answer
    except Exception as e:
        print(f"Error in recipe_qa_chain: {e}")
        return f"Recipe Agent: Error al procesar la consulta de la receta: {e}"

tools = [
    Tool(
        name="InventoryChecker",
        func=check_inventory,
        description="Útil para verificar los ingredientes disponibles y sus cantidades en la base de datos de la cocina. La consulta debe ser sobre los ingredientes específicos."
    ),
    Tool(
        name="RecipeRetriever",
        func=get_recipe_details,
        description="Útil para obtener detalles de una receta, como ingredientes, cantidades necesarias y pasos de preparación, a partir de los documentos PDF. La consulta debe ser el nombre de la receta o una pregunta específica sobre ella."
    ),
]

# --- Supervisor Agent Setup ---
# Supervisor Prompt - Needs careful crafting
# This is a simplified version. A real supervisor would manage a more complex flow.
supervisor_prompt_template = PromptTemplate.from_template("""
Eres un asistente de cocina experto. Tu tarea es determinar si una receta consultada por el usuario puede ser preparada
con los ingredientes disponibles en la cocina.

HERRAMIENTAS DISPONIBLES:
--------------------
{tools}
--------------------

Para responder, sigue este formato rigurosamente:

Thought: [Aquí explicas tu proceso de pensamiento y qué planeas hacer a continuación. Describe los pasos que tomarás para responder la pregunta del usuario. Considera qué herramientas usar, si es necesario.]
Action: [El nombre de la acción a realizar. Debe ser una de [{tool_names}]. Si no necesitas usar una herramienta, usa 'Final Answer' como acción.]
Action Input: [La entrada para la acción. Si la acción es 'Final Answer', esta será tu respuesta final al usuario.]
Observation: [Este es el resultado de la acción. El sistema lo proveerá.]
... (Este patrón de Thought/Action/Action Input/Observation puede repetirse N veces)

Thought: [He llegado a la respuesta final.]
Final Answer: [Tu respuesta final para el usuario. Indicado las herramientas que has usado y el resultado de las mismas.]

Sigue estos pasos para responder al usuario:
1.  Piensa qué información necesitas. ¿Conoces la receta? ¿Conoces los ingredientes de la receta? ¿Conoces el inventario?
2.  Si necesitas información sobre una receta (ingredientes, pasos), usa la herramienta "RecipeRetriever".
3.  Si necesitas verificar los ingredientes disponibles en el inventario, usa la herramienta "InventoryChecker". Debes proporcionar los nombres de los ingredientes que quieres verificar.
4.  Una vez que tengas la lista de ingredientes de la receta y hayas verificado el inventario, analiza si la receta se puede preparar.
5.  Responde al usuario indicando si la receta se puede preparar. Si no se puede, indica claramente qué ingredientes faltan o son insuficientes. Si se puede preparar, simplemente confírmalo.

Comienza!

Pregunta del Usuario: {input}
Thought: {agent_scratchpad}  <!- Asegúrate que agent_scratchpad es el último elemento antes de la respuesta del LLM ->
""")


supervisor_agent = create_react_agent(llm, tools, supervisor_prompt_template)
agent_executor = AgentExecutor(agent=supervisor_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- FastAPI Endpoint ---
class CallRequest(BaseModel):
    prompt: str

@app.post("/call")
async def call_agent(request: CallRequest = Body(...)):
    print(f"Received prompt: {request.prompt}")
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        # For ReAct, the input to invoke should be a dictionary.
        # The specific key expected by create_react_agent might vary slightly or
        # it might directly take the prompt string if the prompt template has a single input variable.
        # Let's assume the template variable is 'input'.
        response = await agent_executor.ainvoke({"input": request.prompt})
        # The actual response from the agent is usually in an 'output' key
        return {"response": response.get("output", "No output from agent")}
    except Exception as e:
        print(f"Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Recipe Agent API is running. Use the /call endpoint to interact with the agent."}

# --- Main Execution / Docker Startup Info ---
if __name__ == "__main__":
    print(f"FastAPI app configured to use OpenAI chat model: {MODEL_NAME} from {OPENAI_API_BASE}")
    print(f"Using OpenAI embeddings model: {OPENAI_EMBEDDINGS_MODEL_NAME}")
    if OPENAI_API_KEY:
        print("OPENAI_API_KEY is set.")
    else:
        print("CRITICAL: OPENAI_API_KEY environment variable is NOT SET. Please set it to use the OpenAI API.")
    print("Ensure your OpenAI account has access to the specified models and sufficient quota.")
    pass # Docker CMD will handle running uvicorn 