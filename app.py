"""Streamlit interface for extracting a PDF knowledge graph into Neo4j.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from html import escape
from pathlib import Path
from typing import Any

import anthropic
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from pyvis.network import Network

from graphlens.extractor import extract, upload_pdf
from graphlens.metadata import build_pdf_metadata
from graphlens.neo4j_loader import KGLoader


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
LOGO_PATH = PROJECT_ROOT / "assets" / "leadso-logo.jpeg"

TYPE_COLORS = {
    "PERSON": "#4e79a7", "ORG": "#f28e2b", "LOCATION": "#59a14f",
    "EVENT": "#edc948", "CONCEPT": "#e15759", "PRODUCT": "#76b7b2",
    "DATE": "#b07aa1", "OTHER": "#bab0ab",
}


def _load_environment() -> None:
    """Load a local .env when present; real environment variables win."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _neo4j_config() -> tuple[str, str, str]:
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", ""),
    )


def _missing_configuration() -> list[str]:
    return [name for name in ("ANTHROPIC_API_KEY", "NEO4J_PASSWORD") if not os.getenv(name)]


def _write_uploaded_pdf(uploaded_file: Any) -> Path:
    """Write the upload to a temporary PDF file and return its path."""
    content = uploaded_file.getvalue()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The PDF is larger than the 100 MB upload limit.")
    if not content.lstrip().startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(content)
        return Path(temp_file.name)


def _fetch_document_graph(uri: str, user: str, password: str, doc_id: str) -> tuple[list[dict], list[dict]]:
    """Read back only the graph relationships created from one PDF."""
    query = """
    MATCH (source:Entity)-[relationship]->(target:Entity)
    WHERE relationship.source_file_id = $doc_id
    RETURN source.name AS source_name, source.type AS source_type,
           source.description AS source_description, target.name AS target_name,
           target.type AS target_type, target.description AS target_description,
           type(relationship) AS predicate, relationship.source_sentence AS evidence,
           relationship.confidence AS confidence, relationship.page AS page
    ORDER BY source_name, predicate, target_name
    LIMIT 500
    """
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            records = [record.data() for record in session.run(query, doc_id=doc_id)]

    nodes: dict[str, dict] = {}
    for record in records:
        nodes[record["source_name"]] = {"name": record["source_name"], "type": record["source_type"] or "OTHER", "description": record["source_description"] or ""}
        nodes[record["target_name"]] = {"name": record["target_name"], "type": record["target_type"] or "OTHER", "description": record["target_description"] or ""}
    return list(nodes.values()), records


def _graph_html(nodes: list[dict], edges: list[dict]) -> str:
    """Build a self-contained, interactive PyVis graph."""
    network = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff")
    network.set_options('{"physics":{"barnesHut":{"gravitationalConstant":-12000},"stabilization":true},"interaction":{"hover":true,"navigationButtons":true}}')
    for node in nodes:
        node_type = node["type"]
        tooltip = f"<b>{escape(node['name'])}</b><br>Type: {escape(node_type)}"
        if node["description"]:
            tooltip += f"<br>{escape(node['description'])}"
        network.add_node(node["name"], label=node["name"], title=tooltip, color=TYPE_COLORS.get(node_type, TYPE_COLORS["OTHER"]))
    for edge in edges:
        details = [f"<b>{escape(edge['predicate'])}</b>"]
        if edge.get("evidence"):
            details.append(escape(edge["evidence"]))
        if edge.get("confidence") is not None:
            details.append(f"Confidence: {edge['confidence']}")
        if edge.get("page") is not None:
            details.append(f"Page: {edge['page']}")
        network.add_edge(edge["source_name"], edge["target_name"], label=edge["predicate"], title="<br>".join(details), arrows="to")
    return network.generate_html(notebook=False)


def _extract_and_load(uploaded_file: Any, domain: str, verify: bool) -> tuple[dict, int, int]:
    """Run the existing PDF extraction pipeline and persist its Neo4j graph."""
    temp_path = _write_uploaded_pdf(uploaded_file)
    client: anthropic.Anthropic | None = None
    remote_file_id: str | None = None
    try:
        metadata = build_pdf_metadata(temp_path)
        client = anthropic.Anthropic()
        remote_file_id = upload_pdf(client, temp_path)
        result = extract(client, remote_file_id, domain=domain.strip() or None, verify=verify, pdf_path=temp_path, source_file_id=metadata["doc_id"])
        result = {"doc_id": metadata["doc_id"], "metadata": metadata, **result}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"{metadata['doc_id']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        uri, user, password = _neo4j_config()
        with KGLoader(uri, user, password) as loader:
            loader.verify_connection()
            node_count, relation_count = loader.load(result)
        return result, node_count, relation_count
    finally:
        if client and remote_file_id:
            try:
                client.beta.files.delete(remote_file_id)
            except Exception:
                pass
        temp_path.unlink(missing_ok=True)


def main() -> None:
    _load_environment()
    st.set_page_config(page_title="Leadso", page_icon="🕸️", layout="wide")
    
    with st.container():
        if LOGO_PATH.is_file():
            st.image(LOGO_PATH, width=140)

    with st.container():
        st.title("Leadso")
        st.caption(
            "Upload a PDF, extract its knowledge graph with Claude, "
            "and inspect the graph stored in Neo4j."
        )
    
    
    #st.title("Leadso")
    #st.caption("Upload a PDF, extract its knowledge graph with Claude, and inspect the graph stored in Neo4j.")
    missing = _missing_configuration()
    if missing:
        st.error("Missing configuration: " + ", ".join(missing) + ". Add these values to .env or your environment.")
        st.code("copy .env.example .env", language="powershell")
        st.stop()

    uri, user, password = _neo4j_config()
    try:
        with KGLoader(uri, user, password) as loader:
            loader.verify_connection()
        st.success(f"Connected to Neo4j at {uri}")
    except (ServiceUnavailable, AuthError) as exc:
        st.error(f"Neo4j connection failed: {exc}")
        st.stop()

    uploaded_file = st.file_uploader("PDF document", type=["pdf"])
    domain = st.text_input("Domain hint (optional)", placeholder="e.g. driving-simulation research")
    verify = st.checkbox("Run verification pass", help="Improves recall but makes a second Claude request.")
    if st.button("Extract and load graph", type="primary", disabled=uploaded_file is None):
        try:
            with st.spinner("Uploading the PDF, extracting its graph, and loading Neo4j…"):
                result, nodes_loaded, relations_loaded = _extract_and_load(uploaded_file, domain, verify)
            st.session_state.update(result=result, nodes_loaded=nodes_loaded, relations_loaded=relations_loaded)
            st.success("Knowledge graph extracted and loaded into Neo4j.")
        except (ValueError, ServiceUnavailable, AuthError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("result")
    if not result:
        return
    st.subheader(f"Results for {result['metadata']['original_filename']}")
    metrics = st.columns(3)
    metrics[0].metric("Entities", len(result["entities"]))
    metrics[1].metric("Relations", len(result["relations"]))
    metrics[2].metric("Document ID", result["doc_id"])
    try:
        nodes, edges = _fetch_document_graph(uri, user, password, result["doc_id"])
        if edges:
            st.subheader("Neo4j graph")
            st.iframe(_graph_html(nodes, edges), height=670)
        else:
            st.warning("Neo4j contains no relations for this document yet.")
    except (ServiceUnavailable, AuthError) as exc:
        st.error(f"Could not read the graph from Neo4j: {exc}")
    st.subheader("Extracted relations")
    st.dataframe(result["relations"], hide_index=True)
    st.download_button("Download KG JSON", data=json.dumps(result, indent=2, ensure_ascii=False), file_name=f"{result['doc_id']}.json", mime="application/json")
    st.caption("Neo4j Browser query for this document:")
    st.code("MATCH (a:Entity)-[r]->(b:Entity)\n" f"WHERE r.source_file_id = '{result['doc_id']}'\n" "RETURN a, r, b", language="cypher")


if __name__ == "__main__":
    main()
