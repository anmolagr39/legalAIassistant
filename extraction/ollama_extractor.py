"""
Local LLM extractor using Ollama
"""
import requests
import json
import logging
from typing import Dict, List, Optional
from extraction.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    CASE_SUMMARY_PROMPT,
    IPC_SECTION_EXTRACTION_PROMPT,
    CONSTITUTION_ARTICLE_PROMPT,
    LEGAL_ACTS_EXTRACTION_PROMPT
)

logger = logging.getLogger(__name__)


class OllamaExtractor:
    """Local LLM extractor using Ollama"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"[OK] Connected to Ollama at {base_url}")
                logger.info(f"[OK] Using model: {model_name}")
            else:
                logger.warning(f"Ollama connection issue: {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
    
    def _call_ollama(self, prompt: str, temperature: float = 0.1) -> Optional[str]:
        """Call Ollama API"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "format": "json"  # Request JSON response
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # 2 min timeout for long responses
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
        
        except requests.exceptions.Timeout:
            logger.error("Ollama request timeout")
            return None
        except Exception as e:
            logger.error(f"Ollama error: {e}")
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
        response = self._call_ollama(prompt)
        
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
        response = self._call_ollama(prompt)
        
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
        response = self._call_ollama(prompt)
        
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
        response = self._call_ollama(prompt)
        
        if response:
            return self._parse_json_response(response)
        return None
    
    def extract_article(self, article_text: str) -> Optional[Dict]:
        """Extract constitutional article information"""
        prompt = CONSTITUTION_ARTICLE_PROMPT.format(text=article_text)
        response = self._call_ollama(prompt)
        
        if response:
            return self._parse_json_response(response)
        return None
    
    def extract_legal_acts(self, text: str) -> Optional[Dict]:
        """Extract legal acts information"""
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        prompt = LEGAL_ACTS_EXTRACTION_PROMPT.format(text=text)
        response = self._call_ollama(prompt)
        
        if response:
            return self._parse_json_response(response)
        return None


def test_ollama_extractor():
    """Test Ollama extractor"""
    print("\n=== Testing Ollama Extractor ===")
    
    extractor = OllamaExtractor()
    
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
    test_ollama_extractor()
