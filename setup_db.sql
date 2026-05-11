-- ============================================================
--  Base de datos: examen_redes_neuronales
--  Tabla: crypto_transactions (20,000 registros)
-- ============================================================

-- Crear la base de datos (ejecutar como superusuario si no existe)
-- Si ya existe, esta línea puede omitirse o comentarse.
CREATE DATABASE examen_redes_neuronales;

-- Conectarse a la base de datos (en psql usa \c examen_redes_neuronales)
\c examen_redes_neuronales

-- Crear la tabla con todos los campos del CSV
CREATE TABLE IF NOT EXISTS crypto_transactions (
    transaction_id              VARCHAR(20)     PRIMARY KEY,
    timestamp                   BIGINT,                        -- Unix timestamp
    blockchain                  VARCHAR(50),
    transaction_type            VARCHAR(50),
    sender_wallet_age_days      INTEGER,
    receiver_wallet_age_days    INTEGER,
    transaction_amount_usd      NUMERIC(15, 2),
    gas_fee_usd                 NUMERIC(10, 4),
    token_type                  VARCHAR(50),
    platform                    VARCHAR(100),
    num_prev_transactions_sender    INTEGER,
    num_prev_transactions_receiver  INTEGER,
    avg_txn_interval_sender_min     NUMERIC(10, 2),
    is_cross_chain              SMALLINT,                      -- 0 o 1
    failed_txn_ratio_sender     NUMERIC(5, 4),
    velocity_score              NUMERIC(8, 4),
    anomaly_score               NUMERIC(8, 4),
    is_scam                     SMALLINT                       -- 0 = legítima, 1 = scam (TARGET)
);

-- Índices útiles para consultas de ML
CREATE INDEX IF NOT EXISTS idx_blockchain       ON crypto_transactions(blockchain);
CREATE INDEX IF NOT EXISTS idx_transaction_type ON crypto_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_is_scam          ON crypto_transactions(is_scam);
CREATE INDEX IF NOT EXISTS idx_token_type       ON crypto_transactions(token_type);
CREATE INDEX IF NOT EXISTS idx_platform         ON crypto_transactions(platform);

-- Confirmar creación
SELECT 'Tabla crypto_transactions creada correctamente.' AS status;
