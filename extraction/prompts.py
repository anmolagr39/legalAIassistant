"""
Prompt templates for LLM-based entity and relation extraction
"""

ENTITY_EXTRACTION_PROMPT = """You are a legal AI assistant specialized in extracting entities from Indian legal documents.

Extract the following entities from the legal text below. Return ONLY valid JSON without any markdown formatting or code blocks.

**Entities to Extract:**
1. **Cases**: Case names and references
2. **Courts**: Court names (Supreme Court, High Court, etc.)
3. **Judges**: Names of judges who delivered judgments
4. **IPC Sections**: Indian Penal Code sections (e.g., Section 302, Section 420)
5. **Articles**: Constitutional articles (e.g., Article 14, Article 21)
6. **Legal Acts**: Names of acts/statutes
7. **Legal Principles**: Legal doctrines, precedents, maxims
8. **Parties**: Names of plaintiffs, defendants, petitioners, respondents
9. **Offenses**: Types of criminal/civil offenses
10. **Punishments**: Types of punishments prescribed
11. **Citations**: Case citations and references to other judgments

**Legal Text:**
{text}

**Output Format:**
Return a JSON object with these exact keys (use empty arrays if no entities found):
{{
  "cases": [list of case names],
  "courts": [list of court names],
  "judges": [list of judge names],
  "ipc_sections": [list of section numbers like "302", "420"],
  "articles": [list of article numbers like "14", "21"],
  "legal_acts": [list of act names],
  "legal_principles": [list of principles/doctrines],
  "parties": [list of party names],
  "offenses": [list of offense types],
  "punishments": [list of punishment types],
  "citations": [list of case citations]
}}

Return only the JSON object, no additional text or formatting.
"""

RELATION_EXTRACTION_PROMPT = """You are a legal AI assistant specialized in identifying relationships between legal entities.

Given the entities extracted from a legal document, identify the relationships between them. Return ONLY valid JSON without any markdown formatting.

**Entities:**
{entities}

**Original Text (for context):**
{text}

**Relationships to Identify:**
1. **CITES**: Case A cites Case B
2. **GOVERNED_BY**: Case governed by IPC Section/Act
3. **APPLIES_TO**: IPC Section applies to Offense
4. **PRESCRIBES**: IPC Section prescribes Punishment
5. **HEARD_IN**: Case heard in Court
6. **DECIDED_BY**: Case decided by Judge
7. **INVOLVES_PARTY**: Case involves Party (plaintiff/defendant)
8. **ESTABLISHES**: Case establishes Legal Principle
9. **REFERS_TO**: Case/Act refers to Constitutional Article
10. **INTERPRETS**: Case interprets Article/Section
11. **FOLLOWS**: Case follows precedent (another case)
12. **OVERRULES**: Case overrules another case

**Output Format:**
Return a JSON object with this exact structure:
{{
  "relationships": [
    {{
      "source": "entity name or identifier",
      "source_type": "Case/IPCSection/Article/etc",
      "target": "entity name or identifier", 
      "target_type": "Case/IPCSection/Article/etc",
      "relation_type": "CITES/GOVERNED_BY/etc",
      "confidence": 0.0-1.0,
      "evidence": "brief quote or evidence from text"
    }}
  ]
}}

Return only the JSON object, no additional text or formatting.
"""

CASE_SUMMARY_PROMPT = """You are a legal AI assistant. Summarize this legal case in structured format. Return ONLY valid JSON without markdown formatting.

**Case Text:**
{text}

**Extract:**
1. Case name and number
2. Court name
3. Date of judgment
4. Judges
5. Main parties (petitioner/respondent)
6. Key facts (2-3 sentences)
7. Legal issues (main questions)
8. Judgment/Holding (decision)
9. Key IPC sections or articles mentioned

**Output Format:**
{{
  "case_name": "...",
  "case_number": "...",
  "court": "...",
  "date": "...",
  "judges": ["..."],
  "petitioner": "...",
  "respondent": "...",
  "facts": "...",
  "issues": ["..."],
  "judgment": "...",
  "sections_mentioned": ["..."],
  "articles_mentioned": ["..."]
}}

Return only the JSON object.
"""

IPC_SECTION_EXTRACTION_PROMPT = """Extract structured information about this IPC section. Return ONLY valid JSON without markdown formatting.

**IPC Section Data:**
{section_data}

**Extract:**
1. Section number
2. Offense description
3. Punishment details
4. Whether cognizable
5. Whether bailable
6. Competent court
7. Related sections (if mentioned)
8. Key terms and concepts

**Output Format:**
{{
  "section_number": "...",
  "offense": "...",
  "punishment": "...",
  "cognizable": "Yes/No",
  "bailable": "Yes/No",
  "court": "...",
  "related_sections": ["..."],
  "key_concepts": ["..."]
}}

Return only the JSON object.
"""

CONSTITUTION_ARTICLE_PROMPT = """Extract structured information about this constitutional article. Return ONLY valid JSON without markdown formatting.

**Article Text:**
{text}

**Extract:**
1. Article number
2. Title/Subject
3. Main provisions
4. Related articles (if mentioned)
5. Key rights or duties
6. Part of constitution

**Output Format:**
{{
  "article_number": "...",
  "title": "...",
  "provisions": "...",
  "related_articles": ["..."],
  "key_points": ["..."],
  "part": "..."
}}

Return only the JSON object.
"""

LEGAL_ACTS_EXTRACTION_PROMPT = """Extract information about legal acts and statutes from this text. Return ONLY valid JSON without markdown formatting.

**Text:**
{text}

**Extract:**
1. Act names
2. Act sections mentioned
3. Year of enactment (if mentioned)
4. Key provisions
5. Related IPC sections or articles

**Output Format:**
{{
  "acts": [
    {{
      "name": "...",
      "sections": ["..."],
      "year": "...",
      "provisions": "...",
      "related_ipc": ["..."],
      "related_articles": ["..."]
    }}
  ]
}}

Return only the JSON object.
"""
