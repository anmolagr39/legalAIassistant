"""
Groq API extractor for legal text processing
"""
import json
import logging
import time
from typing import Dict, List, Optional
from groq import Groq
from extraction.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    CASE_SUMMARY_PROMPT,
    IPC_SECTION_EXTRACTION_PROMPT,
    CONSTITUTION_ARTICLE_PROMPT,
    LEGAL_ACTS_EXTRACTION_PROMPT
)

logger = logging.getLogger(__name__)


class GroqExtractor:
    """Groq API extractor for legal text"""
    
    def __init__(self, api_key: str, api_key_2: str = None, model: str = "llama-3.1-8b-instant"):
        self.api_keys = [api_key]
        if api_key_2:
            self.api_keys.append(api_key_2)
        self.current_key_idx = 0
        self.model = model
        self.client = Groq(api_key=api_key)
        self.temperature = 0.1
        
        logger.info(f"[OK] Initialized Groq model: {model} with {len(self.api_keys)} API key(s)")
    
    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API with retry logic and key rotation"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Rotate to next key if available
                if len(self.api_keys) > 1:
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
                    self.client = Groq(api_key=self.api_keys[self.current_key_idx])
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a legal AI assistant. Extract structured information from legal texts and return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content
            
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                if "429" in error_msg or "rate" in error_msg.lower():
                    logger.warning(f"Rate limit hit, waiting 2s before retry {retry_count}/{max_retries}")
                    time.sleep(2)
                else:
                    logger.warning(f"Groq API error (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} attempts")
                        return None
        
        return None
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON from response"""
        try:
            # Clean markdown if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response: {response_text[:500]}")
            return None
    
    def extract_entities(self, text: str) -> Optional[Dict]:
        """Extract entities from legal text"""
        if not text or len(text.strip()) < 50:
            return None
        
        # Truncate if too long
        max_length = 15000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        response = self._call_groq(prompt)
        
        if response:
            result = self._parse_json_response(response)
            if result:
                logger.info(f"Extracted entities: {sum(len(v) if isinstance(v, list) else 0 for v in result.values())} total")
            return result
        return None
    
    def extract_relations(self, entities: Dict, text: str) -> Optional[Dict]:
        """Extract relationships between entities"""
        if not entities or not text:
            return {"relationships": []}
        
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        entities_str = json.dumps(entities, indent=2)
        prompt = RELATION_EXTRACTION_PROMPT.format(entities=entities_str, text=text)
        response = self._call_groq(prompt)
        
        if response:
            result = self._parse_json_response(response)
            if result and "relationships" in result:
                logger.info(f"Extracted {len(result['relationships'])} relationships")
            return result
        return {"relationships": []}
    
    def summarize_case(self, case_text: str) -> Optional[Dict]:
        """Summarize legal case"""
        max_length = 12000
        if len(case_text) > max_length:
            case_text = case_text[:8000] + "\n...\n" + case_text[-4000:]
        
        prompt = CASE_SUMMARY_PROMPT.format(text=case_text)
        response = self._call_groq(prompt)
        
        if response:
            result = self._parse_json_response(response)
            if result:
                logger.info(f"Summarized case: {result.get('case_name', 'Unknown')}")
            return result
        return None
    
    def extract_ipc_section(self, section_data: Dict) -> Optional[Dict]:
        """Extract structured IPC section information"""
        section_str = json.dumps(section_data, indent=2)
        prompt = IPC_SECTION_EXTRACTION_PROMPT.format(section_data=section_str)
        response = self._call_groq(prompt)
        
        if response:
            return self._parse_json_response(response)
        return None
    
    def extract_article(self, article_text: str) -> Optional[Dict]:
        """Extract constitutional article information"""
        prompt = CONSTITUTION_ARTICLE_PROMPT.format(text=article_text)
        response = self._call_groq(prompt)
        
        if response:
            return self._parse_json_response(response)
        return None
    
    def extract_legal_acts(self, text: str) -> Optional[Dict]:
        """Extract legal acts information"""
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = LEGAL_ACTS_EXTRACTION_PROMPT.format(text=text)
        response = self._call_groq(prompt)
        
        if response:
            return self._parse_json_response(response)
        return None


def test_groq_extractor():
    """Test Groq extractor"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    
    print("\n=== Testing Groq Extractor ===")
    
    extractor = GroqExtractor(api_key=api_key)
    
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
    
    print("\n1. Testing Case Summary...")
    summary = extractor.summarize_case(sample_text)
    if summary:
        print(json.dumps(summary, indent=2))
    
    print("\n2. Testing Entity Extraction...")
    entities = extractor.extract_entities(sample_text)
    if entities:
        print(json.dumps(entities, indent=2))


if __name__ == "__main__":
    test_groq_extractor()
