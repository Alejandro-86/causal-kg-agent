"""Neo4j-backed causal knowledge graph.

Single generic (:Entity {name, type}) node label with a `type` property
(drug/protein/disease/...) rather than one Neo4j label per entity type —
keeps the loader simple since entity types are open-ended (LLM-assigned),
not a fixed small taxonomy. Relationship type in the graph itself IS the
Cypher relationship type (INCREASES, TREATS, ...), which is what makes
multi-hop traversal queries readable.
"""

from neo4j import GraphDatabase

from causal_kg.models import CausalRelation


class Neo4jCausalGraph:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def load_relations(self, relations: list[CausalRelation]) -> None:
        with self._driver.session() as session:
            for rel in relations:
                session.execute_write(self._merge_relation, rel)

    @staticmethod
    def _merge_relation(tx, rel: CausalRelation) -> None:
        rel_type = rel.relation.value.upper()
        tx.run(
            f"""
            MERGE (s:Entity {{name: $subject}})
              ON CREATE SET s.type = $subject_type
            MERGE (o:Entity {{name: $object}})
              ON CREATE SET o.type = $object_type
            MERGE (s)-[r:{rel_type}]->(o)
            SET r.evidence_sentence = $evidence_sentence,
                r.pmid = $pmid,
                r.confidence = $confidence
            """,
            subject=rel.subject,
            subject_type=rel.subject_type,
            object=rel.object,
            object_type=rel.object_type,
            evidence_sentence=rel.evidence_sentence,
            pmid=rel.pmid,
            confidence=rel.confidence,
        )

    def traverse(self, entity_name: str, max_hops: int = 2) -> list[dict]:
        """Find paths up to max_hops from entities whose name contains
        entity_name (case-insensitive), returning each hop's relation
        type, target entity, evidence sentence, and PMID."""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (start:Entity)-[r*1..{max_hops}]->(end:Entity)
                WHERE toLower(start.name) CONTAINS toLower($entity_name)
                RETURN start.name AS start_name,
                       [rel IN relationships(path) | {{
                           type: type(rel),
                           evidence_sentence: rel.evidence_sentence,
                           pmid: rel.pmid,
                           confidence: rel.confidence
                       }}] AS rel_chain,
                       [node IN nodes(path) | node.name] AS node_chain
                LIMIT 20
                """,
                entity_name=entity_name,
            )
            return [dict(record) for record in result]

    def all_nodes_and_edges(self) -> dict:
        """Full graph dump for the webapp's initial visualization."""
        with self._driver.session() as session:
            nodes = session.run(
                "MATCH (n:Entity) RETURN n.name AS name, n.type AS type"
            )
            node_list = [dict(r) for r in nodes]

            edges = session.run(
                """
                MATCH (s:Entity)-[r]->(o:Entity)
                RETURN s.name AS source, o.name AS target, type(r) AS type
                """
            )
            edge_list = [dict(r) for r in edges]

        return {"nodes": node_list, "edges": edge_list}
