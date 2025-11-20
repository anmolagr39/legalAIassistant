"""
Gemini-based entity and relation extractor for legal documents
"""
import google.generativeai as genai
from typing import Dict, List, Optional
import json
import time
import logging
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS
from extraction.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    CASE_SUMMARY_PROMPT,
    IPC_SECTION_EXTRACTION_PROMPT,
    CONSTITUTION_ARTICLE_PROMPT,
    LEGAL_ACTS_EXTRACTION_PROMPT
)

logger = logging.getLogger(__name__)


class GeminiExtractor:
    """Gemini-based entity and relation extractor"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        if not api_key:
            raise ValueError("Gemini API key not provided")
        
        genai.configure(api_key=api_key)
        
        # Initialize model with generation config
        self.generation_config = {
            "temperature": GEMINI_TEMPERATURE,
            "max_output_tokens": GEMINI_MAX_TOKENS,
            "response_mime_type": "application/json",
        }
        
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=self.generation_config
        )
        
        logger.info(f"✓ Initialized Gemini model: {GEMINI_MODEL}")
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean and extract JSON from response"""
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        return response_text.strip()
    
    def _call_gemini(self, prompt: str, retry_count: int = 3) -> Optional[Dict]:
        """Call Gemini API with retry logic"""
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(prompt)
                
                if not response or not response.text:
                    logger.warning(f"Empty response from Gemini (attempt {attempt + 1})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                # Clean and parse JSON
                cleaned_text = self._clean_json_response(response.text)
                result = json.loads(cleaned_text)
                
                return result
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error (attempt {attempt + 1}): {e}")
                logger.debug(f"Response text: {response.text[:500]}")
                
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
            
            except Exception as e:
                logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                
                if "429" in str(e) or "quota" in str(e).lower():
                    # Rate limit - wait longer
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def extract_entities(self, text: str) -> Optional[Dict]:
        """Extract entities from legal text"""
        if not text or len(text.strip()) < 50:
            logger.warning("Text too short for entity extraction")
            return None
        
        # Truncate text if too long (Gemini has context limits)
        max_text_length = 30000  # Conservative limit
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
            logger.info(f"Truncated text to {max_text_length} characters")
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        result = self._call_gemini(prompt)
        
        if result:
            logger.info(f"Extracted entities: {sum(len(v) if isinstance(v, list) else 0 for v in result.values())} total")
        
        return result
    
    def extract_relations(self, entities: Dict, text: str) -> Optional[Dict]:
        """Extract relationships between entities"""
        if not entities or not text:
            return {"relationships": []}
        
        # Truncate text if too long
        max_text_length = 20000
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        entities_str = json.dumps(entities, indent=2)
        prompt = RELATION_EXTRACTION_PROMPT.format(entities=entities_str, text=text)
        result = self._call_gemini(prompt)
        
        if result and "relationships" in result:
            logger.info(f"Extracted {len(result['relationships'])} relationships")
        
        return result
    
    def summarize_case(self, case_text: str) -> Optional[Dict]:
        """Summarize legal case"""
        max_text_length = 25000
        if len(case_text) > max_text_length:
            # Take beginning and end of case
            case_text = case_text[:15000] + "\n...\n" + case_text[-10000:]
        
        prompt = CASE_SUMMARY_PROMPT.format(text=case_text)
        result = self._call_gemini(prompt)
        
        if result:
            logger.info(f"Summarized case: {result.get('case_name', 'Unknown')}")
        
        return result
    
    def extract_ipc_section(self, section_data: Dict) -> Optional[Dict]:
        """Extract structured IPC section information"""
        section_str = json.dumps(section_data, indent=2)
        prompt = IPC_SECTION_EXTRACTION_PROMPT.format(section_data=section_str)
        result = self._call_gemini(prompt)
        
        return result
    
    def extract_article(self, article_text: str) -> Optional[Dict]:
        """Extract constitutional article information"""
        prompt = CONSTITUTION_ARTICLE_PROMPT.format(text=article_text)
        result = self._call_gemini(prompt)
        
        return result
    
    def extract_legal_acts(self, text: str) -> Optional[Dict]:
        """Extract legal acts information"""
        max_text_length = 20000
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        prompt = LEGAL_ACTS_EXTRACTION_PROMPT.format(text=text)
        result = self._call_gemini(prompt)
        
        return result
    
    def batch_extract(self, texts: List[str], extract_type: str = "entities", 
                     batch_size: int = 5, delay: float = 1.0) -> List[Optional[Dict]]:
        """Batch extract from multiple texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            for text in batch:
                if extract_type == "entities":
                    result = self.extract_entities(text)
                elif extract_type == "summary":
                    result = self.summarize_case(text)
                else:
                    result = None
                
                results.append(result)
                time.sleep(delay)  # Rate limiting
            
            # Longer pause between batches
            if i + batch_size < len(texts):
                time.sleep(2.0)
        
        return results


def test_gemini_extractor():
    """Test Gemini extractor"""
    extractor = GeminiExtractor()
    
    # Test with sample legal text
    sample_text = """
    Masud Khan v State Of Uttar Pradesh
    Supreme Court of India
    26 September 1973
    Writ Petition No. 117 of 1973
    
    The petitioner was arrested under Section 14 of the Foreigners Act. 
    The case involves Article 14 of the Constitution. The Supreme Court 
    held that the burden of proof lies on the petitioner under Section 9 
    of the Foreigners Act.
    """
    
    print("\n=== Testing Entity Extraction ===")
    entities = extractor.extract_entities(sample_text)
    if entities:
        print(json.dumps(entities, indent=2))
    
    print("\n=== Testing Relation Extraction ===")
    if entities:
        relations = extractor.extract_relations(entities, sample_text)
        if relations:
            print(json.dumps(relations, indent=2))
    
    print("\n=== Testing Case Summary ===")
    summary = extractor.summarize_case(sample_text)
    if summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    test_gemini_extractor()
