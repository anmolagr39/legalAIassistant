// Neo4j Browser Queries for Legal Knowledge Graph
// Open Neo4j Browser at http://localhost:7474 and paste these queries

// ============================================================================
// 1. VIEW MORE NODES (increase limit)
// ============================================================================

// View 100 cases with relationships
MATCH (n)
RETURN n
LIMIT 100

// View 200 nodes
MATCH (n)
RETURN n
LIMIT 200

// View all IPC sections (445 total)
MATCH (s:IPCSection)
RETURN s
LIMIT 500


// ============================================================================
// 2. VIEW SPECIFIC SUBGRAPHS
// ============================================================================

// Cases with IPC Section 302 (Murder)
MATCH path = (c:Case)-[:GOVERNED_BY]->(s:IPCSection {section_number: "302"})
RETURN path
LIMIT 50

// Cases with judges and courts
MATCH path = (c:Case)-[:PRESIDED_BY]->(j:Judge),
             (c)-[:HEARD_IN]->(court:Court)
RETURN path
LIMIT 50

// IPC sections with their offenses and punishments
MATCH path = (s:IPCSection)-[:PRESCRIBES]->(o:Offense),
             (s)-[:PRESCRIBES]->(p:Punishment)
RETURN path
LIMIT 100


// ============================================================================
// 3. STATISTICS QUERIES
// ============================================================================

// Count all node types
MATCH (n)
RETURN labels(n)[0] as NodeType, count(*) as Count
ORDER BY Count DESC

// Count all relationship types
MATCH ()-[r]->()
RETURN type(r) as RelationType, count(*) as Count
ORDER BY Count DESC

// Top 10 judges by case count
MATCH (j:Judge)<-[:PRESIDED_BY]-(c:Case)
RETURN j.name as Judge, count(c) as CaseCount
ORDER BY CaseCount DESC
LIMIT 10

// Top 10 most cited IPC sections
MATCH (s:IPCSection)<-[:GOVERNED_BY]-(c:Case)
RETURN s.section_number as Section, s.offense as Offense, count(c) as CaseCount
ORDER BY CaseCount DESC
LIMIT 10


// ============================================================================
// 4. SEARCH QUERIES
// ============================================================================

// Search cases by keyword
MATCH (c:Case)
WHERE toLower(c.case_name) CONTAINS "murder"
RETURN c
LIMIT 50

// Find cases between date range
MATCH (c:Case)
WHERE c.date >= "2020-01-01" AND c.date <= "2023-12-31"
RETURN c
LIMIT 50

// Cases citing specific article
MATCH (c:Case)-[:REFERS_TO]->(a:Article)
WHERE a.article_number = "21"
RETURN c, a
LIMIT 50


// ============================================================================
// 5. NETWORK ANALYSIS
// ============================================================================

// Citation network (if cases cite each other)
MATCH path = (c1:Case)-[:CITES]->(c2:Case)
RETURN path
LIMIT 100

// Connected subgraph: Case -> Judge -> Other Cases
MATCH path = (c1:Case)-[:PRESIDED_BY]->(j:Judge)<-[:PRESIDED_BY]-(c2:Case)
WHERE id(c1) < id(c2)
RETURN path
LIMIT 50


// ============================================================================
// 6. FILTER BY SPECIFIC ENTITIES
// ============================================================================

// All cases from a specific year
MATCH (c:Case)
WHERE c.date STARTS WITH "2022"
RETURN c
LIMIT 100

// Cases involving bailable offenses
MATCH (c:Case)-[:GOVERNED_BY]->(s:IPCSection)
WHERE s.bailable = "Yes"
RETURN c, s
LIMIT 50

// Cases in Supreme Court
MATCH (c:Case)-[:HEARD_IN]->(court:Court)
WHERE court.name CONTAINS "Supreme"
RETURN c, court
LIMIT 100


// ============================================================================
// 7. CHANGE VISUALIZATION SETTINGS
// ============================================================================

// In Neo4j Browser, you can:
// 1. Click the gear icon (⚙) at bottom left
// 2. Set "Initial Node Display" to higher value (e.g., 300, 500, 1000)
// 3. Or run queries with higher LIMIT values

// To see ALL 1482 cases at once (may be slow):
MATCH (n)
RETURN n
LIMIT 2000


// ============================================================================
// 8. EXPLORE SPECIFIC RELATIONSHIPS
// ============================================================================

// Cases and their complete context
MATCH (c:Case)
OPTIONAL MATCH (c)-[r1:GOVERNED_BY]->(s:IPCSection)
OPTIONAL MATCH (c)-[r2:PRESIDED_BY]->(j:Judge)
OPTIONAL MATCH (c)-[r3:HEARD_IN]->(court:Court)
OPTIONAL MATCH (c)-[r4:REFERS_TO]->(a:Article)
RETURN c, r1, s, r2, j, r3, court, r4, a
LIMIT 50
