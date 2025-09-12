-- ================================================
-- CREACIÓN DE LA VISTA MATERIALIZADA
-- ================================================

CREATE MATERIALIZED VIEW law_frame.legal_search_combined AS
SELECT
  -- Campos del item
  li.id                     AS item_id,
  li.law_id                 AS law_identifier_ref,
  li.second_id,
  li.title                  AS item_title,
  li.content,
  li.url                    AS item_url,
  li.deprecated             AS item_deprecated,
  -- Campos de la ley
  l.id                      AS law_id,
  l.identifier              AS law_identifier,
  l.title                   AS law_title,
  l.url_html_consolidated   AS law_url,
  l.deprecated              AS law_deprecated,
  -- Vector combinado para búsquedas en español
  (
    -- títulos
    setweight(to_tsvector('spanish', COALESCE(li.title, '')), 'A') ||
    setweight(to_tsvector('spanish', COALESCE(l.title, '')),  'A') ||
    -- identificador del artículo
    setweight(to_tsvector('spanish', COALESCE(li.second_id, '')), 'B') ||
    -- primeras 5 palabras del contenido
    setweight(
      to_tsvector(
        'spanish',
        array_to_string(
          (regexp_split_to_array(COALESCE(li.content,''), E'\\s+'))[1:5],
          ' '
        )
      ),
      'A'
    ) ||
    -- cuerpo (primeros 800 chars)
    setweight(to_tsvector('spanish', COALESCE(substring(li.content, 1, 800), '')), 'D')
  ) AS combined_search_vector,
  li.updated_at,
  l.last_update              AS law_last_update,
  li.chapter,
  'law_item'::text           AS result_type
FROM law_frame.law_items li
JOIN law_frame.laws l 
  ON li.law_id::text = l.identifier;

-- ============================
-- ÍNDICES PARA LA VISTA
-- ============================
CREATE UNIQUE INDEX IF NOT EXISTS legal_search_combined_item_id_uidx
  ON law_frame.legal_search_combined (item_id);

CREATE INDEX IF NOT EXISTS legal_search_combined_vec_idx
  ON law_frame.legal_search_combined
  USING GIN (combined_search_vector);

-- ============================
-- FUNCIÓN DE REFRESCO
-- ============================
CREATE OR REPLACE FUNCTION law_frame.refresh_legal_search_view()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY law_frame.legal_search_combined;
    EXCEPTION 
        WHEN OTHERS THEN
            REFRESH MATERIALIZED VIEW law_frame.legal_search_combined;
    END;
END;
$$;

-- Programar refresco automático con pg_cron (cada 2 días a las 02:00)
SELECT cron.schedule(
    'refresh-legal-search', 
    '0 2 */2 * *',
    'SELECT law_frame.refresh_legal_search_view();'
);

-- Refresco inicial manual
SELECT law_frame.refresh_legal_search_view();

-- Prueba de búsqueda
SELECT * FROM law_frame.search_legal_content('art 2 Código Civil ', 10, FALSE, NULL::INTEGER);

-- ============================
-- PERMISOS
-- ============================
GRANT USAGE ON SCHEMA law_frame TO service_role;
GRANT SELECT ON ALL TABLES IN SCHEMA law_frame TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA law_frame TO service_role;
GRANT SELECT ON law_frame.legal_search_combined TO service_role;
GRANT EXECUTE ON FUNCTION law_frame.refresh_legal_search_view() TO service_role;

-- ===================================================
-- FUNCIÓN DE BÚSQUEDA AVANZADA: search_legal_content
-- ===================================================
-- Nota: requiere la extensión "unaccent"
--   CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE OR REPLACE FUNCTION law_frame.search_legal_content(
    p_search_query        text,
    p_limit_count         integer DEFAULT 20,
    p_include_deprecated  boolean DEFAULT FALSE,
    p_law_id              integer DEFAULT NULL,
    p_min_rank            real    DEFAULT 0.01,
    p_full_content        boolean DEFAULT TRUE
)
RETURNS TABLE(
    item_id        bigint,
    law_id         integer,
    second_id      varchar(255),
    item_title     text,
    content        text,
    url            text,
    law_identifier text,
    law_title      text,
    search_rank    real,
    match_type     text,
    content_length integer,
    is_truncated   boolean
)
LANGUAGE plpgsql
AS $$
DECLARE
  q_es       tsquery;
  q_simple   tsquery;
  q_unaccent tsquery;  -- Query sin acentos (fallback)
  nums       text[];
  has_num    boolean;
BEGIN
  -- Validaciones de parámetros
  IF p_search_query IS NULL OR btrim(p_search_query) = '' THEN
    RAISE EXCEPTION 'p_search_query no puede ser vacío';
  END IF;
  IF p_limit_count <= 0 OR p_limit_count > 100000 THEN
    RAISE EXCEPTION 'p_limit_count debe estar entre 1 y 100000, recibido: %', p_limit_count;
  END IF;
  IF p_min_rank < 0.0 OR p_min_rank > 1.0 THEN
    RAISE EXCEPTION 'p_min_rank debe estar entre 0.0 y 1.0, recibido: %', p_min_rank;
  END IF;

  -- Construcción de queries
  q_es       := websearch_to_tsquery('spanish', p_search_query);
  q_simple   := websearch_to_tsquery('simple',  p_search_query);
  q_unaccent := websearch_to_tsquery('simple', unaccent(p_search_query));

  -- Extraer números para buscar artículos (ej: "art 2")
  SELECT array_agg(m[1]) INTO nums
  FROM regexp_matches(p_search_query, '\d+', 'g') m;
  has_num := nums IS NOT NULL AND array_length(nums,1) > 0;

  -- ====================
  -- BÚSQUEDA PRINCIPAL
  -- ====================
  RETURN QUERY
  WITH items AS (
    -- Items individuales
    ...
  ),
  items_ranked AS (
    -- Items con ranking
    ...
  ),
  laws AS (
    -- Leyes completas
    ...
  ),
  laws_ranked AS (
    -- Leyes con ranking
    ...
  ),
  ranked AS (
    SELECT * FROM items_ranked
    UNION ALL
    SELECT * FROM laws_ranked
  )
  SELECT
    r.item_id,
    r.law_id,
    r.second_id,
    r.item_title,
    r.content,
    r.url,
    r.law_identifier,
    r.law_title,
    r.search_rank,
    r.match_type,
    r.content_length,
    r.is_truncated
  FROM ranked r
  WHERE r.search_rank_col >= p_min_rank
  ORDER BY
    CASE WHEN r.match_type_col = 'article' THEN 1 ELSE 2 END,
    r.search_rank_col DESC,
    r.content_length_col DESC
  LIMIT p_limit_count;
END;
$$;

-- ===========================================
-- COMENTARIOS DE DOCUMENTACIÓN (resumen)
-- ===========================================
/*
MEJORAS PRINCIPALES:
1. p_full_content=TRUE: evita truncados por defecto
2. p_min_rank configurable: filtra ruido irrelevante
3. Validación de parámetros robusta
4. Metadatos adicionales: content_length e is_truncated
5. Priorización explícita de artículos vs leyes
*/

-- ============================
-- MÁS PERMISOS
-- ============================
GRANT USAGE ON SCHEMA law_frame TO service_role;
GRANT SELECT ON law_frame.legal_search_combined TO service_role;
GRANT EXECUTE ON FUNCTION law_frame.search_legal_content(text,integer,boolean,integer,real,boolean) TO service_role;

-- Permisos para usuarios autenticados
GRANT USAGE ON SCHEMA law_frame TO authenticated;
GRANT SELECT ON law_frame.legal_search_combined TO authenticated;
GRANT EXECUTE ON FUNCTION law_frame.search_legal_content(text,integer,boolean,integer,real,boolean) TO authenticated;

-- ============================
-- EJEMPLO DE USO
-- ============================
SELECT * FROM law_frame.search_legal_content('artículo 2 código civil');
