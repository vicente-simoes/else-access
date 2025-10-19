-- one simple read-only query
SELECT abalance FROM pgbench_accounts WHERE aid = random_between(1, 1000000);