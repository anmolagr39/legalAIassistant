"""
Legal case queries specifically designed for Past Cases RAG system evaluation.
"""

from typing import List

def get_legal_case_queries() -> List[str]:
    """
    Return 10 natural question-type queries that users would typically ask about legal cases.
    These questions simulate real user queries for more realistic retrieval testing.
    """
    return [
        "What are the key Supreme Court judgments on right to life under Article 21?",
        "How can I file a writ petition for violation of fundamental rights?", 
        "What is the procedure for criminal appeal in High Court and Supreme Court?",
        "Which cases established the principle of equality before law under Article 14?",
        "What are the reasonable restrictions on freedom of speech and expression?",
        "How do courts apply principles of natural justice in administrative matters?",
        "When can administrative decisions be challenged through judicial review?",
        "What remedies are available in civil cases for breach of contract?",
        "How are government service matters like promotion and seniority decided by courts?",
        "What is the constitutional validity of President's rule under Article 356?"
    ]