"""
Legal queries for ROUGE-C evaluation of Past Cases RAG system.
Contains 10 realistic user questions about legal cases and procedures relevant to the database.
"""

from typing import List

def get_simple_legal_queries() -> List[str]:
    """
    Return 10 legal case queries relevant to the actual database content.
    Based on Supreme Court cases in areas like citizenship, rent control, securities, etc.
    """
    return [
        "What are the Supreme Court decisions on citizenship and nationality disputes?",
        "How do courts determine whether a person is a foreigner under the Foreigners Act?", 
        "What are the key rulings on rent control and tenant eviction laws?",
        "How do courts handle cases involving negotiable instruments and cheque bounce?",
        "What is the Supreme Court's approach to issue estoppel in criminal proceedings?",
        "How do courts deal with demolition and reconstruction cases under rent acts?",
        "What are the landmark judgments on special courts and their jurisdiction?",
        "How do courts interpret constitutional validity of state legislation?",
        "What remedies are available for illegal detention and habeas corpus petitions?",
        "How do Supreme Court decisions handle securities transaction related offences?"
    ]