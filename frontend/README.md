
# Writing Prototypes (Frontend)

This is a TypeScript/React rewrite of the original Python/Streamlit prototype based on `app.py`.

## Setup

1.  Make sure you have Node.js installed.
2.  Install dependencies:
    ```bash
    npm install
    ```

## Running

1.  Start the development server:
    ```bash
    npm run dev
    ```
2.  Open your browser at `http://localhost:5173`.

## Backend

This frontend is configured to communicate with the existing backend at `https://tools.kenarnold.org/api`.
If you want to run the local backend (`custom_llm.py`), you need to:
1.  Run the python backend (with GPU if available):
    ```bash
    python custom_llm.py --gpu
    ```
2.  Update `src/api.ts` to point to `http://localhost:8000` (or whatever port FastAPI runs on, usually 8000).

## Components

-   `Rewrite`: Corresponding to `rewrite_with_predictions` in python.
-   `Highlights`: Corresponding to `highlight_edits` or `get_highlights` in python.
-   `TypeAssistant`: Corresponding to `type_assistant_response` in python.
