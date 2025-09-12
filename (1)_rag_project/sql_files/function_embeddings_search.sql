-- ================================================
-- FUNCIÓN: search_similar_law_items
-- ================================================
-- Esta función busca artículos de leyes (law_items) similares
-- a un embedding de entrada usando la extensión "pgvector".
-- 
-- Requiere:
--   CREATE EXTENSION IF NOT EXISTS vector;
--
-- Tipo de dato usado: halfvec (vector comprimido)
-- Operador usado: <=> (distancia coseno)
-- ================================================

-- Borrar versión anterior si existiera
DROP FUNCTION IF EXISTS search_similar_law_items(halfvec, integer, double precision, boolean);

-- Crear función
CREATE OR REPLACE FUNCTION search_similar_law_items(
    p_query_embedding halfvec,          -- embedding de búsqueda
    p_limit_count integer,              -- número máximo de resultados
    p_similarity_threshold double precision, -- umbral mínimo de similitud coseno
    p_include_deprecated boolean        -- incluir registros deprecated o no
)
RETURNS TABLE(
    -- =====================
    -- Campos de law_items
    -- =====================
    item_id integer,
    law_id text,
    second_id text,
    item_title text,
    content text,
    url text,
    num_questions integer,
    max_number_questions integer,
    chapter_info jsonb,
    vector_data halfvec,
    embedding_model text,
    item_updated_at timestamp,
    item_deprecated boolean,
    
    -- =====================
    -- Campos de laws
    -- =====================
    law_identifier text,
    law_title text,
    last_update timestamp,
    diario text,
    scope jsonb,
    department jsonb,
    rank jsonb,
    disposition_date timestamp,
    publication_date timestamp,
    validity_date timestamp,
    expired_validity boolean,
    url_eli text,
    url_html_consolidated text,
    law_deprecated boolean,
    
    -- =====================
    -- Campos calculados
    -- =====================
    cosine_similarity float,  -- similitud coseno (0-1)
    law_unic_id integer       -- id interno único de la ley
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        -- law_items
        li.id,
        li.law_id,
        li.second_id,
        li.title,
        li.content,
        li.url,
        li.num_questions,
        li.max_number_questions,
        li.chapter,
        li.vector,
        li.embedding_model,
        li.updated_at,
        li.deprecated,
        
        -- laws
        l.identifier,
        l.title,
        l.last_update,
        l.diario,
        l.scope,
        l.department,
        l.rank,
        l.disposition_date,
        l.publication_date,
        l.validity_date,
        l.expired_validity,
        l.url_eli,
        l.url_html_consolidated,
        l.deprecated,
        
        -- Cálculo de similitud coseno
        -- Nota: pgvector devuelve distancia coseno (0=igual, 1=ortogonal).
        --       Aquí la transformamos a similitud: 1 - distancia
        (1 - (li.vector <=> p_query_embedding))::float AS cosine_similarity,
        l.id AS law_unic_id
        
    FROM law_frame.law_items li
    LEFT JOIN law_frame.laws l 
      ON li.law_id = l.identifier
    WHERE 
        li.vector IS NOT NULL
        AND (1 - (li.vector <=> p_query_embedding)) >= p_similarity_threshold
        AND (
            p_include_deprecated = true 
            OR (
                (li.deprecated IS NULL OR li.deprecated = false)
                AND (l.deprecated IS NULL OR l.deprecated = false)
                AND (l.expired_validity IS NULL OR l.expired_validity = false)
            )
        )
    ORDER BY 
        li.vector <=> p_query_embedding  -- ordenar por menor distancia = más similar
    LIMIT p_limit_count;
END;
$$;
