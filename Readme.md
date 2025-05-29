# Multi Agent Solution Recipe 🍷

Here you will find and agentic solution to read multiple sources in this case we use MogoDB and a Knowledge Base loaded from PDFs to a vector store. The Supervisor will call receipt to know the ingredients and mongoDB to know which of those are available. Only set your OPENAI_API_KEY on docker compose and build.

## Tools Used
* FastAPI
* Langchain
* Docker
* OpenAI (Agents, Embeddings)
* Gemini 2.5 + Cursor

## Service Curl

```
curl --location 'localhost:8000/call' \
--header 'Content-Type: application/json' \
--data '{
    "prompt":"Puedo hacer pasta carbonara?"
}'
```

## Diagram

```mermaid
graph TD
    subgraph "User Interaction"
        User["External User"]
    end

    subgraph "Application Components (within fastapi_app service)"
        APIServer["FastAPI /call Endpoint"]
        SupervisorAgent["Supervisor Agent (Langchain)"]
        ToolCheckInventory["Tool: check_inventory"]
        ToolGetRecipe["Tool: get_recipe_details"]
        RecipeKnowledgeBase["Recipe Knowledge Base (Vector Store from PDFs)"]
        MongoDB["MongoDB (cocina_db.ingredientes_disponibles)"]
    end

    subgraph "Persistent Data Sources"
        RecipePDFs["Recipe PDFs in /documentos"]
        AvailableIngredientsCSV["available_ingredients.csv in /documentos"]
    end

    User -->|"HTTP POST with recipe query"| APIServer;
    APIServer -->|"Passes query to"| SupervisorAgent;
    SupervisorAgent -->|"Calls Tool"| ToolCheckInventory;
    ToolCheckInventory -->|"Reads current inventory"| MongoDB;
    SupervisorAgent -->|"Calls Tool"| ToolGetRecipe;
    ToolGetRecipe -->|"Retrieves recipe steps/ingredients"| RecipeKnowledgeBase;

    RecipeKnowledgeBase --"Indexes content from"--> RecipePDFs;
    MongoDB --"Initialized by"--> AvailableIngredientsCSV;

    classDef user fill:#ffdfba,stroke:#333,stroke-width:2px,color:#000;
    class User user;
    classDef app fill:#cce5ff,stroke:#333,stroke-width:2px,color:#000;
    class APIServer,SupervisorAgent,ToolCheckInventory,ToolGetRecipe,RecipeKnowledgeBase app;
    classDef service fill:#e6ffe6,stroke:#333,stroke-width:2px,color:#000;
    class MongoDB, service;
    classDef datastyle fill:#lightgrey,stroke:#333,stroke-width:2px,color:#FFF;
    class RecipePDFs,AvailableIngredientsCSV datastyle;
``` 