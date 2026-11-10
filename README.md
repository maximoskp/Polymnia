# Polymnia
AI agents for document processing, retrieval and question-answering. This demo includes:

- *Polymnia_admin*: Answers questions regarding administrative processes described by the documents in this page: [Οδηγός Διοικητικών Διαδικασιών](https://hmu.gr/odigos-dioikitikon-diadikasion/)

Demo chat questions:

```
Ποια είναι η διαδικασία για την έγκριση μετακίνησης ενός μέλους ΔΕΠ;

Και ποια δικαιολογητικά χρειάζονται;

Ποιοί είναι κάποιοι πιθανοί λόγοι απόρριψης της αίτησης;
```

To run it: `uvicorn app_admin.main:app --host 0.0.0.0 --port 4142 --reload`

- *Polymnia_student*: Answers questions regarding student-related issues described in the regulatory documents of the department.

Demo chat questions: TODO

How to run it: `uvicorn app_students.main:app --host 0.0.0.0 --port 4143 --reload`

Notes:

python -m pip freeze > requirements.txt
