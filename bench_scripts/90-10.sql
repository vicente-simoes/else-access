-- one simple read-only query
\set aid (1 + random() * 1000000)::int
SELECT abalance FROM pgbench_accounts WHERE aid = :aid;
