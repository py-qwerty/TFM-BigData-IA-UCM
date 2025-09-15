-- ================================================
-- CREACIÓN DE TABLA LAWS (TODAS LAS LEYES DEL BOE CONSOLIDADO)
-- ================================================

create table law_frame.laws (
  id serial not null,
  identifier text not null,
  title text null,
  last_update timestamp without time zone null,
  diario character varying(255) null,
  scope jsonb null,
  department jsonb null,
  rank jsonb null,
  disposition_date timestamp without time zone null,
  official_number character varying(255) null,
  publication_date timestamp without time zone null,
  diario_number character varying(255) null,
  validity_date timestamp without time zone null,
  expired_validity boolean null,
  consolidation_state jsonb null,
  url_eli text null,
  url_html_consolidated text null,
  chapter bigint[] null,
  chapter_check boolean null,
  deprecated boolean null,
  search_vector tsvector GENERATED ALWAYS as (
    (
      (
        setweight(
          to_tsvector('spanish'::regconfig, COALESCE(title, ''::text)),
          'A'::"char"
        ) || setweight(
          to_tsvector(
            'spanish'::regconfig,
            COALESCE(identifier, ''::text)
          ),
          'B'::"char"
        )
      ) || setweight(
        to_tsvector(
          'spanish'::regconfig,
          (
            COALESCE(official_number, (''::text)::character varying)
          )::text
        ),
        'C'::"char"
      )
    )
  ) STORED null,
  constraint laws_pkey1 primary key (id),
  constraint laws_identifier_key unique (identifier)
) TABLESPACE pg_default;

create index IF not exists idx_laws_title_gin on law_frame.laws using gin (to_tsvector('spanish'::regconfig, title)) TABLESPACE pg_default;

create index IF not exists idx_laws_identifier_gin on law_frame.laws using gin (to_tsvector('spanish'::regconfig, identifier)) TABLESPACE pg_default;

create index IF not exists idx_laws_identifier_text on law_frame.laws using btree (identifier text_pattern_ops) TABLESPACE pg_default;

create index IF not exists idx_laws_search_vector on law_frame.laws using gin (search_vector) TABLESPACE pg_default;

create trigger trigger_update_law_items
after
update on law_frame.laws for EACH row
execute FUNCTION law_frame.trigger_for_update_law_items_deprecated ();


-- ================================================
-- CREACIÓN DE TABLA LAW ITEMS (ARTÍCULOS,PREAMBULOS,...)
-- ================================================


create table law_frame.law_items (
  id bigint not null,
  law_id character varying(255) not null,
  second_id character varying(255) not null,
  title text null,
  updated_at timestamp without time zone null,
  url text null,
  content text null,
  num_questions integer null,
  max_number_questions integer null,
  chapter jsonb null,
  vector extensions.halfvec null,
  deprecated boolean null,
  embedding_model text null,
  search_vector tsvector GENERATED ALWAYS as (
    (
      (
        (
          setweight(
            to_tsvector('spanish'::regconfig, COALESCE(title, ''::text)),
            'A'::"char"
          ) || setweight(
            to_tsvector(
              'spanish'::regconfig,
              (COALESCE(law_id, (''::text)::character varying))::text
            ),
            'B'::"char"
          )
        ) || setweight(
          to_tsvector(
            'spanish'::regconfig,
            (
              COALESCE(second_id, (''::text)::character varying)
            )::text
          ),
          'C'::"char"
        )
      ) || setweight(
        to_tsvector(
          'spanish'::regconfig,
          COALESCE("substring" (content, 1, 500), ''::text)
        ),
        'D'::"char"
      )
    )
  ) STORED null,
  constraint law_items_pkey primary key (id),
  constraint law_items_lawid_secondid_unique unique (law_id, second_id),
  constraint law_items_tmp_url_key unique (url)
) TABLESPACE pg_default;

create index IF not exists law_items_vector_idx on law_frame.law_items using hnsw (vector extensions.halfvec_l2_ops)
with
  (m = '16', ef_construction = '200') TABLESPACE pg_default;

create index IF not exists idx_law_items_title_gin on law_frame.law_items using gin (to_tsvector('spanish'::regconfig, title)) TABLESPACE pg_default;

create index IF not exists idx_law_items_content_gin on law_frame.law_items using gin (to_tsvector('spanish'::regconfig, content)) TABLESPACE pg_default;

create index IF not exists idx_law_items_title_text on law_frame.law_items using btree (title text_pattern_ops) TABLESPACE pg_default;

create index IF not exists idx_law_items_content_partial on law_frame.law_items using btree ("substring" (content, 1, 500) text_pattern_ops) TABLESPACE pg_default;

create index IF not exists law_items_vector_cosine_idx on law_frame.law_items using hnsw (vector extensions.halfvec_cosine_ops)
with
  (m = '16', ef_construction = '200') TABLESPACE pg_default;

create index IF not exists idx_law_items_search_vector on law_frame.law_items using gin (search_vector) TABLESPACE pg_default;

create trigger trigger_set_deprecated BEFORE INSERT
or
update on law_frame.law_items for EACH row
execute FUNCTION law_frame.set_deprecated_on_derogado ();