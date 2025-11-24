"""
Generate Ground Truth Dataset for Legal Assistant System Evaluation
Based on actual ingested data from ChromaDB and Neo4j KG
"""
import json
import os
from datetime import datetime

def create_ground_truth_dataset():
    """Create 30 ground truth questions with answers based on ingested data"""
    
    ground_truth_data = {
        "metadata": {
            "created_date": datetime.now().isoformat(),
            "description": "Ground truth dataset for evaluating Legal Assistant RAG system with LLM as judge",
            "total_questions": 30,
            "data_sources": {
                "ipc_rag": "933 documents from FIR/IPC dataset",
                "past_cases_rag": "1068 case law documents from Supreme Court",
                "knowledge_graph": "1482 Cases, 439 IPC Sections, 189 Judges, 16 Articles in Neo4j",
                "legal_acts_rag": "Not currently ingested - empty collection"
            },
            "question_categories": {
                "ipc_sections": 8,
                "case_law": 8,
                "knowledge_graph": 10,
                "cross_domain": 4
            }
        },
        "questions": []
    }
    
    # ====================================================================================
    # IPC RAG QUESTIONS (8 questions)
    # ====================================================================================
    
    ipc_questions = [
        {
            "id": 1,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "What is IPC Section 140 and what is the punishment for violating it?",
            "ground_truth_answer": "IPC Section 140 deals with wearing the dress or carrying any token used by a soldier, sailor or airman with intent that it may be believed that he is such a soldier, sailor or airman. The punishment is imprisonment of either description for a term which may extend to three months, or with fine which may extend to five hundred rupees, or with both. It is a Cognizable and Bailable offense tried by Any Magistrate.",
            "query_type": "factual_retrieval",
            "difficulty": "easy",
            "expected_sources": ["IPC Section 140 document"]
        },
        {
            "id": 2,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "What is the punishment for receiving property taken by war or depredation mentioned in IPC sections 125 and 126?",
            "ground_truth_answer": "According to IPC Section 127, whoever receives any property knowing the same to have been taken in the commission of offences mentioned in sections 125 and 126, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine and to forfeiture of the property so received. It is a Cognizable and Non-Bailable offense tried by Court of Session.",
            "query_type": "factual_retrieval",
            "difficulty": "medium",
            "expected_sources": ["IPC Section 127 document"]
        },
        {
            "id": 3,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "What happens if a public servant voluntarily allows a prisoner of State or war to escape?",
            "ground_truth_answer": "According to IPC Section 128, a public servant having the custody of any State prisoner or prisoner of war who voluntarily allows such prisoner to escape from any place in which such prisoner is confined, shall be punished with imprisonment for life, or imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine. It is a Cognizable and Non-Bailable offense tried by Court of Session.",
            "query_type": "factual_retrieval",
            "difficulty": "medium",
            "expected_sources": ["IPC Section 128 document"]
        },
        {
            "id": 4,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "What is the difference between IPC Section 128 and 129 regarding prisoner escape?",
            "ground_truth_answer": "IPC Section 128 deals with a public servant voluntarily allowing a prisoner to escape, which is punished with imprisonment for life or up to 10 years plus fine (Cognizable, Non-Bailable, tried by Court of Session). IPC Section 129 deals with a public servant negligently suffering such prisoner to escape, which is punished with simple imprisonment for up to 3 years plus fine (Cognizable, Bailable, tried by Magistrate First Class). The key difference is 'voluntary' vs 'negligent' action, resulting in significantly different penalties.",
            "query_type": "comparison",
            "difficulty": "hard",
            "expected_sources": ["IPC Section 128 document", "IPC Section 129 document"]
        },
        {
            "id": 5,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "What constitutes aiding escape of a State prisoner under IPC Section 130?",
            "ground_truth_answer": "IPC Section 130 covers whoever knowingly aids or assists any State prisoner or prisoner of war in escaping from lawful custody, or rescues or attempts to rescue any such prisoner, or harbours or conceals any such prisoner who has escaped from lawful custody, or offers or attempts to offer any resistance to the recapture of such prisoner. The punishment is imprisonment for life, or with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine. It is a Cognizable and Non-Bailable offense tried by Court of Session.",
            "query_type": "factual_retrieval",
            "difficulty": "medium",
            "expected_sources": ["IPC Section 130 document"]
        },
        {
            "id": 6,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "What is the offense under IPC Section 131 and its punishment?",
            "ground_truth_answer": "IPC Section 131 deals with abetting mutiny, or attempting to seduce an officer, soldier, sailor or airman from his allegiance or duty in the Army, Navy or Air Force of the Government of India. The punishment is imprisonment for life, or with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine. It is a Cognizable and Non-Bailable offense tried by Court of Session.",
            "query_type": "factual_retrieval",
            "difficulty": "easy",
            "expected_sources": ["IPC Section 131 document"]
        },
        {
            "id": 7,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "What is the enhanced punishment under IPC Section 132 compared to Section 131?",
            "ground_truth_answer": "IPC Section 132 deals with abetment of mutiny, if mutiny is actually committed in consequence of that abetment. While Section 131 (abetting mutiny without it occurring) is punished with imprisonment for life or up to 10 years plus fine, Section 132 (when mutiny is actually committed) can be punished with death or with imprisonment for life, or imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine. The key difference is the death penalty option when mutiny actually occurs. Both are Cognizable and Non-Bailable offenses tried by Court of Session.",
            "query_type": "comparison",
            "difficulty": "hard",
            "expected_sources": ["IPC Section 131 document", "IPC Section 132 document"]
        },
        {
            "id": 8,
            "category": "ipc_sections",
            "data_source": "IPC RAG (ChromaDB)",
            "question": "Is IPC Section 133 a bailable or non-bailable offense and which court has jurisdiction?",
            "ground_truth_answer": "IPC Section 133 deals with abetment of an assault by an officer, soldier, sailor or airman on his superior officer when in the execution of his office. It is a Cognizable and Non-Bailable offense tried by Magistrate First Class. The punishment is imprisonment of either description for a term which may extend to three years, and shall also be liable to fine.",
            "query_type": "factual_retrieval",
            "difficulty": "medium",
            "expected_sources": ["IPC Section 133 document"]
        },
    ]
    
    # ====================================================================================
    # PAST CASES RAG QUESTIONS (8 questions)
    # ====================================================================================
    
    case_law_questions = [
        {
            "id": 9,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "What was the main issue in Masud Khan v State Of Uttar Pradesh case?",
            "ground_truth_answer": "The main issue in Masud Khan v State Of Uttar Pradesh (1973) was whether the petitioner, who claimed to be an Indian citizen, was illegally arrested and confined under Paragraph 5 of the Foreigners (Internment) Order, 1962. The petitioner had come to India from Pakistan on a Pakistani passport and visa, and the central question was whether he could establish his Indian citizenship by proving he was present in India on January 26, 1950. The Supreme Court held that under Section 9 of the Foreigners Act, the burden of proving that a person is not a foreigner lies upon him, and the petitioner failed to discharge this burden.",
            "query_type": "case_analysis",
            "difficulty": "medium",
            "expected_sources": ["Masud Khan v State Of Uttar Pradesh case document"]
        },
        {
            "id": 10,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "What is the principle of issue estoppel as discussed in Masud Khan v State Of Uttar Pradesh?",
            "ground_truth_answer": "In Masud Khan v State Of Uttar Pradesh, the Supreme Court discussed issue estoppel referring to Pritam Singh v State of Punjab and Sambasivam v Public Prosecutor. The Court held that issue estoppel arises only if both the earlier and subsequent proceedings were criminal prosecutions. The principle is that where an issue of fact has been tried by a competent court and a finding has been reached in favor of an accused, such finding constitutes an estoppel against the prosecution in subsequent criminal trials for a different offense. However, in Masud Khan's case, the present proceeding was merely an action under the Foreigners (Internment) Order for deportation, not a criminal prosecution, so issue estoppel did not apply.",
            "query_type": "legal_principle",
            "difficulty": "hard",
            "expected_sources": ["Masud Khan v State Of Uttar Pradesh case document"]
        },
        {
            "id": 11,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "What were the key issues in Prabhakaran Nair v State Of Tamil Nadu And Ors (1987)?",
            "ground_truth_answer": "The key legal issues in Prabhakaran Nair v State Of Tamil Nadu And Ors (1987) were: (1) The constitutional validity of Section 14(1)(b) and Section 16(2) of the Tamil Nadu Buildings (Lease and Rent Control) Act, 1960, which allows landlords to seek eviction for demolition and reconstruction without providing for re-induction of tenants after reconstruction; (2) Whether the absence of a provision for tenant re-induction after reconstruction is arbitrary and violative of Article 14 of the Constitution; (3) Whether Section 16(2), which exempts reconstructed buildings from rent control for 5 years, is discriminatory. The Supreme Court upheld the validity of these provisions, holding that the distinction between repairs (where tenant re-induction is provided) and reconstruction (where it is not) is rational.",
            "query_type": "case_analysis",
            "difficulty": "hard",
            "expected_sources": ["Prabhakaran Nair v State Of Tamil Nadu And Ors case document"]
        },
        {
            "id": 12,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "Which judge delivered the judgment in Hiten P. Dalal v Bratindranath Banerjee case?",
            "ground_truth_answer": "The judgment in Hiten P. Dalal v Bratindranath Banerjee (2001) was delivered by Justice Ruma Pal, J.",
            "query_type": "factual_retrieval",
            "difficulty": "easy",
            "expected_sources": ["Hiten P. Dalal v Bratindranath Banerjee case document"]
        },
        {
            "id": 13,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "What was the main legal question regarding jurisdiction in Hiten P. Dalal v Bratindranath Banerjee?",
            "ground_truth_answer": "The main legal question in Hiten P. Dalal v Bratindranath Banerjee was whether the Special Court established under the Special Court (Trial of Offences relating to Transactions in Securities) Act, 1992 had jurisdiction to try offences under Section 138 of the Negotiable Instruments Act when the offence (dishonoured cheques) occurred after June 6, 1992. The court held that the Special Court had jurisdiction because the period specified in Section 3(2) of the Act ('after 1 April 1991 and on or before 6 June 1992') qualifies the word 'transactions' not 'offence'. Since the cheques were issued during the statutory period for transactions in securities that occurred during that period, the Special Court had jurisdiction even though the cheques were dishonored later.",
            "query_type": "legal_principle",
            "difficulty": "hard",
            "expected_sources": ["Hiten P. Dalal v Bratindranath Banerjee case document"]
        },
        {
            "id": 14,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "What court heard the case of Masud Khan v State Of Uttar Pradesh?",
            "ground_truth_answer": "Masud Khan v State Of Uttar Pradesh was heard by the Supreme Court of India on 26 September 1973. It was a Writ Petition No. 117 of 1973.",
            "query_type": "factual_retrieval",
            "difficulty": "easy",
            "expected_sources": ["Masud Khan v State Of Uttar Pradesh case document"]
        },
        {
            "id": 15,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "What are the three presumptions available under the Negotiable Instruments Act discussed in Hiten P. Dalal case?",
            "ground_truth_answer": "In Hiten P. Dalal v Bratindranath Banerjee, the Supreme Court discussed three presumptions under the Negotiable Instruments Act: (1) Section 118 presumes that every negotiable instrument was made or drawn for consideration; (2) Section 138 provides that where a cheque is dishonored due to insufficient funds, the drawer shall be deemed to have committed an offence (subject to conditions of presentation, notice, and non-payment after notice); (3) Section 139 presumes that the holder of a cheque received it for the discharge, in whole or in part, of any debt or other liability. These are mandatory presumptions ('shall presume'), placing the evidential burden on the accused to prove the contrary.",
            "query_type": "legal_principle",
            "difficulty": "hard",
            "expected_sources": ["Hiten P. Dalal v Bratindranath Banerjee case document"]
        },
        {
            "id": 16,
            "category": "case_law",
            "data_source": "Past Cases RAG (ChromaDB)",
            "question": "What was the date of judgment in Prabhakaran Nair v State Of Tamil Nadu case?",
            "ground_truth_answer": "The judgment in Prabhakaran Nair, Etc. v State Of Tamil Nadu And Ors. was delivered on 3 September 1987 by the Supreme Court of India.",
            "query_type": "factual_retrieval",
            "difficulty": "easy",
            "expected_sources": ["Prabhakaran Nair v State Of Tamil Nadu And Ors case document"]
        },
    ]
    
    # ====================================================================================
    # KNOWLEDGE GRAPH QUESTIONS (10 questions)
    # ====================================================================================
    
    kg_questions = [
        {
            "id": 17,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "How many IPC Sections are stored in the knowledge graph?",
            "ground_truth_answer": "There are 439 IPC Sections stored in the Neo4j knowledge graph.",
            "query_type": "statistical",
            "difficulty": "easy",
            "expected_sources": ["Neo4j node count query"]
        },
        {
            "id": 18,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "How many case documents are indexed in the knowledge graph?",
            "ground_truth_answer": "There are 1,482 cases indexed in the Neo4j knowledge graph.",
            "query_type": "statistical",
            "difficulty": "easy",
            "expected_sources": ["Neo4j node count query"]
        },
        {
            "id": 19,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "How many judges are recorded in the knowledge graph?",
            "ground_truth_answer": "There are 189 judges recorded in the Neo4j knowledge graph.",
            "query_type": "statistical",
            "difficulty": "easy",
            "expected_sources": ["Neo4j node count query"]
        },
        {
            "id": 20,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "What types of relationships exist in the knowledge graph and which is most common?",
            "ground_truth_answer": "The knowledge graph contains the following relationship types: CITES (1,184 relationships - most common), APPLIES_TO (379), PRESCRIBES (379), HEARD_IN (300), and DECIDED_BY (286). The CITES relationship, which represents cases citing other cases or legal provisions, is the most prevalent.",
            "query_type": "statistical_analysis",
            "difficulty": "medium",
            "expected_sources": ["Neo4j relationship count query"]
        },
        {
            "id": 21,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "Which court's cases are primarily indexed in the knowledge graph?",
            "ground_truth_answer": "The Supreme Court of India is the primary court whose cases are indexed in the knowledge graph. There is 1 Court node in the graph representing the Supreme Court of India.",
            "query_type": "factual_retrieval",
            "difficulty": "easy",
            "expected_sources": ["Neo4j Court nodes query"]
        },
        {
            "id": 22,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "How many Constitutional Articles are stored in the knowledge graph?",
            "ground_truth_answer": "There are 16 Constitutional Articles stored in the Neo4j knowledge graph.",
            "query_type": "statistical",
            "difficulty": "easy",
            "expected_sources": ["Neo4j node count query"]
        },
        {
            "id": 23,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "List some of the judges who have presided over cases in the knowledge graph.",
            "ground_truth_answer": "Some of the judges recorded in the knowledge graph include: A. Alagiriswami J., Sabyasachi Mukharji J., Ruma Pal J., S. B. Sinha J., K. N. Wanchoo J., R. V. Raveendran J., Kailas Nath Wanchoo J., B. P. Sinha J., D. A. Desai J., and P. Venkatarama Reddi J., among 189 total judges.",
            "query_type": "factual_retrieval",
            "difficulty": "medium",
            "expected_sources": ["Neo4j Judge nodes query"]
        },
        {
            "id": 24,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "What is the total number of nodes in the knowledge graph?",
            "ground_truth_answer": "The knowledge graph contains a total of 2,586 nodes, comprising 1,482 Cases, 439 IPC Sections, 378 Offenses, 189 Judges, 81 Punishments, 16 Articles, and 1 Court.",
            "query_type": "statistical",
            "difficulty": "medium",
            "expected_sources": ["Neo4j node count aggregation"]
        },
        {
            "id": 25,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "What types of nodes exist in the knowledge graph?",
            "ground_truth_answer": "The knowledge graph contains the following node types: Case (1,482 nodes), IPCSection (439 nodes), Offense (378 nodes), Judge (189 nodes), Punishment (81 nodes), Article (16 nodes), and Court (1 node).",
            "query_type": "schema_understanding",
            "difficulty": "medium",
            "expected_sources": ["Neo4j node labels query"]
        },
        {
            "id": 26,
            "category": "knowledge_graph",
            "data_source": "Neo4j Knowledge Graph",
            "question": "What Constitutional Articles are mentioned in the knowledge graph?",
            "ground_truth_answer": "The knowledge graph contains several Constitutional Articles including: Article 3 (Implementation of Article 2), Article 1 (Enclaves), Article 2 (Enclaves), Article 99 (Parliamentary privileges and procedure), Article 39 (Right to Constitutional Remedies), Article 109 (Bills and Assent procedures), Article 132 (Officers and servants of Supreme Court), Article 124 (Appointment of High Court Judges), Article 239 (Power of administrator to promulgate Ordinances), and Article 363 (Territorial waters and continental shelf), among 16 total articles.",
            "query_type": "factual_retrieval",
            "difficulty": "medium",
            "expected_sources": ["Neo4j Article nodes query"]
        },
    ]
    
    # ====================================================================================
    # CROSS-DOMAIN QUESTIONS (4 questions)
    # ====================================================================================
    
    cross_domain_questions = [
        {
            "id": 27,
            "category": "cross_domain",
            "data_source": "IPC RAG + Knowledge Graph",
            "question": "What are some IPC sections related to military offenses based on the available data?",
            "ground_truth_answer": "Based on the IPC RAG and knowledge graph data, several IPC sections relate to military offenses: Section 131 (Abetting mutiny or attempting to seduce military personnel from allegiance), Section 132 (Abetment of mutiny if committed), Section 133 (Abetment of assault by military personnel on superior officer), Section 134 (Abetment of such assault if committed), Section 135 (Abetment of desertion of military personnel), and Section 140 (Wearing military dress or token with fraudulent intent). These sections protect military discipline and order.",
            "query_type": "synthesis",
            "difficulty": "hard",
            "expected_sources": ["IPC RAG documents", "Neo4j IPC Section nodes"]
        },
        {
            "id": 28,
            "category": "cross_domain",
            "data_source": "Past Cases RAG + Knowledge Graph",
            "question": "Which judges have delivered notable judgments in the cases stored in the system?",
            "ground_truth_answer": "Based on the past cases RAG and knowledge graph, notable judges who have delivered judgments include: A. Alagiriswami J. (in Masud Khan v State Of Uttar Pradesh, 1973), Sabyasachi Mukharji J. (in Prabhakaran Nair v State Of Tamil Nadu, 1987), and Ruma Pal J. (in Hiten P. Dalal v Bratindranath Banerjee, 2001). The knowledge graph contains a total of 189 judges who have presided over the 1,482 cases in the system.",
            "query_type": "synthesis",
            "difficulty": "medium",
            "expected_sources": ["Past Cases RAG documents", "Neo4j Judge nodes"]
        },
        {
            "id": 29,
            "category": "cross_domain",
            "data_source": "All Sources",
            "question": "What types of legal information can be queried from this legal assistant system?",
            "ground_truth_answer": "The legal assistant system provides access to multiple types of legal information: (1) IPC Sections with detailed descriptions, offenses, punishments, and procedural information (933 documents covering 439 sections in the knowledge graph); (2) Supreme Court case law with case details, judgments, legal reasoning, and judicial opinions (1,068 documents covering 1,482 cases); (3) Knowledge graph relationships connecting cases, IPC sections, judges, courts, constitutional articles, offenses, and punishments; (4) Constitutional Articles (16 articles); (5) Information about 189 judges and their case associations. The system does not currently have legal acts data ingested despite having infrastructure for it.",
            "query_type": "system_capabilities",
            "difficulty": "medium",
            "expected_sources": ["All data sources"]
        },
        {
            "id": 30,
            "category": "cross_domain",
            "data_source": "Knowledge Graph + Past Cases RAG",
            "question": "How are cases connected to IPC sections in the knowledge graph?",
            "ground_truth_answer": "In the knowledge graph, cases are connected to IPC sections through relationship types including GOVERNED_BY (cases governed by specific IPC sections), APPLIES_TO (379 relationships showing IPC sections applying to specific contexts), and CITES (1,184 relationships for citations). The graph contains 1,482 cases and 439 IPC sections. While the system tracks these relationships, the specific case-IPC mappings depend on the entities extracted from the case documents. The APPLIES_TO and PRESCRIBES relationships (379 each) connect IPC sections to offenses and punishments, creating a comprehensive legal knowledge structure.",
            "query_type": "relationship_analysis",
            "difficulty": "hard",
            "expected_sources": ["Neo4j relationship queries", "Case documents"]
        },
    ]
    
    # Combine all questions
    all_questions = ipc_questions + case_law_questions + kg_questions + cross_domain_questions
    ground_truth_data["questions"] = all_questions
    
    return ground_truth_data

def main():
    """Generate and save ground truth dataset"""
    print("Generating Ground Truth Dataset...")
    print("=" * 80)
    
    # Generate dataset
    dataset = create_ground_truth_dataset()
    
    # Create output directory
    output_dir = "ground_truth_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    output_file = os.path.join(output_dir, "legal_assistant_ground_truth.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Create a CSV version for easy viewing
    import csv
    csv_file = os.path.join(output_dir, "legal_assistant_ground_truth.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Category', 'Data Source', 'Question', 'Ground Truth Answer', 'Query Type', 'Difficulty'])
        
        for q in dataset['questions']:
            writer.writerow([
                q['id'],
                q['category'],
                q['data_source'],
                q['question'],
                q['ground_truth_answer'],
                q['query_type'],
                q['difficulty']
            ])
    
    # Print summary
    print(f"\n✅ Ground Truth Dataset Created Successfully!")
    print(f"   Total Questions: {len(dataset['questions'])}")
    print(f"\n📊 Question Distribution:")
    print(f"   - IPC Sections: {len([q for q in dataset['questions'] if q['category'] == 'ipc_sections'])}")
    print(f"   - Case Law: {len([q for q in dataset['questions'] if q['category'] == 'case_law'])}")
    print(f"   - Knowledge Graph: {len([q for q in dataset['questions'] if q['category'] == 'knowledge_graph'])}")
    print(f"   - Cross-Domain: {len([q for q in dataset['questions'] if q['category'] == 'cross_domain'])}")
    
    print(f"\n📁 Output Files:")
    print(f"   - JSON: {output_file}")
    print(f"   - CSV: {csv_file}")
    
    print(f"\n🎯 Difficulty Levels:")
    for difficulty in ['easy', 'medium', 'hard']:
        count = len([q for q in dataset['questions'] if q['difficulty'] == difficulty])
        print(f"   - {difficulty.capitalize()}: {count}")
    
    print("\n" + "=" * 80)
    print("Dataset ready for LLM-as-a-Judge evaluation!")

if __name__ == "__main__":
    main()
