# qp-ai-assessment
Problem Statement - Contextual Chat Bot

To test your Flask application using Postman, follow these steps:

### 1. **Starting the Application**
   - Ensure your virtual environment is activated.
   - Run your Flask application using the shell script:
     ```bash
     chmod +x run_app.sh
     ./run_app.sh
     ```
   - This will start the Flask server. Note the port number (usually `8501`).

### 2. **Setting Up Postman**
   - Open Postman.
   - Enter the API endpoint (for example, `http://127.0.0.1:8501/upload_documents`) in the request URL field.

### 3. **Testing the `/upload_documents` Endpoint**
   - ![upload_documents_endpoint.png](https://github.com/iusaidmulla/qp-ai-assessment/blob/50d42f0df8abf26bae7d7ef4cddf26cf8d307cd7/upload_documents_endpoint.png)
   - Set the HTTP method to `POST`.
   - Select `Body` → `form-data`.
     - **File Upload:** Add files to test the document upload feature:
       - For the key, set it as `files` (as in your code: `request.files.getlist("files")`).
       - Click the "Select Files" button to upload PDF, DOCX, or TXT files.
     - **URL Input (optional):** If you want to test the URL loading feature:
       - Add a JSON object under the `Body` tab, and select `form-data` → `JSON`.
       - Example:
         ```
         Key : files
         Value : Select File; Get the option to upload file from local machine.
         ```
   - Click **Send** to upload the document and get the response.

### 4. **Testing the `/ask_pdf` Endpoint**
   - ![https://github.com/iusaidmulla/qp-ai-assessment/blob/849c47dd622657146e30379e252270f57d7378d9/ask_pdf_endpoint.png](https://github.com/iusaidmulla/qp-ai-assessment/blob/129fe224633bba1d6a627a5136760d9593aff088/ask_pdf_endpoint.png)
   - Set the HTTP method to `POST`.
   - In the `Body` tab, choose `raw` and select `JSON`.
   - Enter the following JSON to ask a question about the uploaded documents:
     ```json
     {
       "query": "Tell me about the document"
     }
     ```
   - Click **Send** and check the response, which should include the answer and the relevant sources.

### 5. **Testing the `/performance_report` Endpoint**
   - ![https://github.com/iusaidmulla/qp-ai-assessment/blob/9ac05fabb5ce88b2ac7f67b09b8faad0ad316f06/performance_report_endpoint.png](https://github.com/iusaidmulla/qp-ai-assessment/blob/aaa1bfca395928e51c2e41aa743523a5bf6248e7/performance_report_endpoint.png)
   - Set the HTTP method to `GET`.
   - Enter `http://127.0.0.1:8501/performance_report` in the URL field.
   - Click **Send** to retrieve the performance report.

### 6. **Testing the `/delete_vector_db` Endpoint**
   - Set the HTTP method to `POST`.
   - In the `Body` tab, select `raw` → `JSON`.
   - Add the password required to delete the database:
     ```json
     {
       "password": "your_password_here"
     }
     ```
   - Click **Send** to attempt the deletion of the vector database.

These are the basic steps to interact with your endpoints using Postman!
