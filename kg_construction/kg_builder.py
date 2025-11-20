"""
Neo4j Knowledge Graph Schema and Builder
"""
from typing import Dict, List, Optional
from config.neo4j_config import Neo4jConnection
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Build knowledge graph in Neo4j"""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        self.conn = neo4j_conn
    
    def create_ipc_section(self, section_data: Dict) -> bool:
        """Create IPC Section node"""
        query = """
        MERGE (s:IPCSection {section_number: $section_number})
        SET s.description = $description,
            s.offense = $offense,
            s.punishment = $punishment,
            s.cognizable = $cognizable,
            s.bailable = $bailable,
            s.court = $court,
            s.url = $url,
            s.updated_at = datetime()
        RETURN s
        """
        
        try:
            self.conn.execute_write(query, section_data)
            return True
        except Exception as e:
            logger.error(f"Error creating IPC section {section_data.get('section_number')}: {e}")
            return False
    
    def create_case(self, case_data: Dict) -> bool:
        """Create Case node"""
        query = """
        MERGE (c:Case {case_id: $case_id})
        SET c.case_name = $case_name,
            c.case_number = $case_number,
            c.date = $date,
            c.content = $content,
            c.updated_at = datetime()
        RETURN c
        """
        
        try:
            self.conn.execute_write(query, case_data)
            
            # Create court relationship if court exists
            if case_data.get('court'):
                self.create_case_court_relationship(
                    case_data['case_id'],
                    case_data['court']
                )
            
            return True
        except Exception as e:
            logger.error(f"Error creating case {case_data.get('case_id')}: {e}")
            return False
    
    def create_court(self, court_name: str) -> bool:
        """Create Court node"""
        query = """
        MERGE (c:Court {name: $name})
        SET c.updated_at = datetime()
        RETURN c
        """
        
        try:
            self.conn.execute_write(query, {"name": court_name})
            return True
        except Exception as e:
            logger.error(f"Error creating court {court_name}: {e}")
            return False
    
    def create_judge(self, judge_name: str) -> bool:
        """Create Judge node"""
        query = """
        MERGE (j:Judge {name: $name})
        SET j.updated_at = datetime()
        RETURN j
        """
        
        try:
            self.conn.execute_write(query, {"name": judge_name})
            return True
        except Exception as e:
            logger.error(f"Error creating judge {judge_name}: {e}")
            return False
    
    def create_article(self, article_data: Dict) -> bool:
        """Create Constitutional Article node"""
        query = """
        MERGE (a:Article {article_number: $article_number})
        SET a.title = $title,
            a.content = $content,
            a.provisions = $provisions,
            a.part = $part,
            a.updated_at = datetime()
        RETURN a
        """
        
        try:
            self.conn.execute_write(query, article_data)
            return True
        except Exception as e:
            logger.error(f"Error creating article {article_data.get('article_number')}: {e}")
            return False
    
    def create_legal_act(self, act_data: Dict) -> bool:
        """Create Legal Act node"""
        query = """
        MERGE (a:LegalAct {name: $name})
        SET a.year = $year,
            a.provisions = $provisions,
            a.updated_at = datetime()
        RETURN a
        """
        
        try:
            self.conn.execute_write(query, act_data)
            return True
        except Exception as e:
            logger.error(f"Error creating legal act {act_data.get('name')}: {e}")
            return False
    
    def create_offense(self, offense_type: str) -> bool:
        """Create Offense node"""
        query = """
        MERGE (o:Offense {offense_type: $offense_type})
        SET o.updated_at = datetime()
        RETURN o
        """
        
        try:
            self.conn.execute_write(query, {"offense_type": offense_type})
            return True
        except Exception as e:
            logger.error(f"Error creating offense {offense_type}: {e}")
            return False
    
    def create_punishment(self, punishment_type: str) -> bool:
        """Create Punishment node"""
        query = """
        MERGE (p:Punishment {punishment_type: $punishment_type})
        SET p.updated_at = datetime()
        RETURN p
        """
        
        try:
            self.conn.execute_write(query, {"punishment_type": punishment_type})
            return True
        except Exception as e:
            logger.error(f"Error creating punishment {punishment_type}: {e}")
            return False
    
    def create_party(self, party_data: Dict) -> bool:
        """Create Party node"""
        query = """
        MERGE (p:Party {name: $name})
        SET p.role = $role,
            p.updated_at = datetime()
        RETURN p
        """
        
        try:
            self.conn.execute_write(query, party_data)
            return True
        except Exception as e:
            logger.error(f"Error creating party {party_data.get('name')}: {e}")
            return False
    
    def create_legal_principle(self, principle: str) -> bool:
        """Create Legal Principle node"""
        query = """
        MERGE (lp:LegalPrinciple {principle: $principle})
        SET lp.updated_at = datetime()
        RETURN lp
        """
        
        try:
            self.conn.execute_write(query, {"principle": principle})
            return True
        except Exception as e:
            logger.error(f"Error creating legal principle: {e}")
            return False
    
    # Relationship creation methods
    
    def create_case_court_relationship(self, case_id: str, court_name: str) -> bool:
        """Create HEARD_IN relationship between Case and Court"""
        # First ensure court exists
        self.create_court(court_name)
        
        query = """
        MATCH (c:Case {case_id: $case_id})
        MATCH (court:Court {name: $court_name})
        MERGE (c)-[r:HEARD_IN]->(court)
        RETURN r
        """
        
        try:
            self.conn.execute_write(query, {
                "case_id": case_id,
                "court_name": court_name
            })
            return True
        except Exception as e:
            logger.error(f"Error creating case-court relationship: {e}")
            return False
    
    def create_case_judge_relationship(self, case_id: str, judge_name: str) -> bool:
        """Create DECIDED_BY relationship between Case and Judge"""
        self.create_judge(judge_name)
        
        query = """
        MATCH (c:Case {case_id: $case_id})
        MATCH (j:Judge {name: $judge_name})
        MERGE (c)-[r:DECIDED_BY]->(j)
        RETURN r
        """
        
        try:
            self.conn.execute_write(query, {
                "case_id": case_id,
                "judge_name": judge_name
            })
            return True
        except Exception as e:
            logger.error(f"Error creating case-judge relationship: {e}")
            return False
    
    def create_case_ipc_relationship(self, case_id: str, section_number: str) -> bool:
        """Create GOVERNED_BY relationship between Case and IPC Section"""
        query = """
        MATCH (c:Case {case_id: $case_id})
        MATCH (s:IPCSection {section_number: $section_number})
        MERGE (c)-[r:GOVERNED_BY]->(s)
        RETURN r
        """
        
        try:
            self.conn.execute_write(query, {
                "case_id": case_id,
                "section_number": section_number
            })
            return True
        except Exception as e:
            logger.error(f"Error creating case-IPC relationship: {e}")
            return False
    
    def create_case_article_relationship(self, case_id: str, article_number: str) -> bool:
        """Create REFERS_TO relationship between Case and Article"""
        query = """
        MATCH (c:Case {case_id: $case_id})
        MATCH (a:Article {article_number: $article_number})
        MERGE (c)-[r:REFERS_TO]->(a)
        RETURN r
        """
        
        try:
            self.conn.execute_write(query, {
                "case_id": case_id,
                "article_number": article_number
            })
            return True
        except Exception as e:
            logger.error(f"Error creating case-article relationship: {e}")
            return False
    
    def create_case_citation_relationship(self, case_id: str, cited_case_name: str) -> bool:
        """Create CITES relationship between Cases"""
        query = """
        MATCH (c1:Case {case_id: $case_id})
        MERGE (c2:Case {case_name: $cited_case_name})
        MERGE (c1)-[r:CITES]->(c2)
        RETURN r
        """
        
        try:
            self.conn.execute_write(query, {
                "case_id": case_id,
                "cited_case_name": cited_case_name
            })
            return True
        except Exception as e:
            logger.error(f"Error creating citation relationship: {e}")
            return False
    
    def create_ipc_offense_relationship(self, section_number: str, offense_type: str) -> bool:
        """Create APPLIES_TO relationship between IPC Section and Offense"""
        self.create_offense(offense_type)
        
        query = """
        MATCH (s:IPCSection {section_number: $section_number})
        MATCH (o:Offense {offense_type: $offense_type})
        MERGE (s)-[r:APPLIES_TO]->(o)
        RETURN r
        """
        
        try:
            self.conn.execute_write(query, {
                "section_number": section_number,
                "offense_type": offense_type
            })
            return True
        except Exception as e:
            logger.error(f"Error creating IPC-offense relationship: {e}")
            return False
    
    def create_ipc_punishment_relationship(self, section_number: str, punishment_type: str) -> bool:
        """Create PRESCRIBES relationship between IPC Section and Punishment"""
        self.create_punishment(punishment_type)
        
        query = """
        MATCH (s:IPCSection {section_number: $section_number})
        MATCH (p:Punishment {punishment_type: $punishment_type})
        MERGE (s)-[r:PRESCRIBES]->(p)
        RETURN r
        """
        
        try:
            self.conn.execute_write(query, {
                "section_number": section_number,
                "punishment_type": punishment_type
            })
            return True
        except Exception as e:
            logger.error(f"Error creating IPC-punishment relationship: {e}")
            return False
    
    def create_generic_relationship(self, source_id: str, source_type: str,
                                   target_id: str, target_type: str,
                                   relation_type: str, properties: Dict = None) -> bool:
        """Create a generic relationship between any two nodes"""
        query = f"""
        MATCH (s:{source_type})
        WHERE s.case_id = $source_id OR s.section_number = $source_id 
           OR s.article_number = $source_id OR s.name = $source_id
        MATCH (t:{target_type})
        WHERE t.case_id = $target_id OR t.section_number = $target_id 
           OR t.article_number = $target_id OR t.name = $target_id
        MERGE (s)-[r:{relation_type}]->(t)
        """
        
        if properties:
            set_clause = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            query += f" SET {set_clause}"
        
        query += " RETURN r"
        
        params = {
            "source_id": source_id,
            "target_id": target_id,
            **(properties or {})
        }
        
        try:
            self.conn.execute_write(query, params)
            return True
        except Exception as e:
            logger.error(f"Error creating {relation_type} relationship: {e}")
            return False


def test_kg_builder():
    """Test KG builder"""
    from config.neo4j_config import Neo4jConnection
    
    conn = Neo4jConnection()
    conn.connect()
    
    builder = KnowledgeGraphBuilder(conn)
    
    # Test creating nodes
    print("\n=== Testing Node Creation ===")
    
    # IPC Section
    builder.create_ipc_section({
        "section_number": "302",
        "description": "Punishment for murder",
        "offense": "Murder",
        "punishment": "Death or Life Imprisonment",
        "cognizable": "Yes",
        "bailable": "No",
        "court": "Court of Session",
        "url": "https://example.com"
    })
    print("✓ Created IPC Section")
    
    # Case
    builder.create_case({
        "case_id": "TEST_001",
        "case_name": "Test Case v. State",
        "case_number": "Crl. Appeal 123/2020",
        "date": "2020-01-15",
        "court": "Supreme Court of India",
        "content": "Test case content"
    })
    print("✓ Created Case")
    
    # Article
    builder.create_article({
        "article_number": "14",
        "title": "Equality before law",
        "content": "Test content",
        "provisions": "Equal protection",
        "part": "Part III"
    })
    print("✓ Created Article")
    
    # Test relationships
    print("\n=== Testing Relationship Creation ===")
    builder.create_case_ipc_relationship("TEST_001", "302")
    print("✓ Created Case-IPC relationship")
    
    conn.close()


if __name__ == "__main__":
    test_kg_builder()
