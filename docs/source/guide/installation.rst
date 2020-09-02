Installation
==========================
These scripts can be downloaded from

| https://github.com/ooovector/qtlab_replacement

| https://github.com/ooovector/qsweepy-notebooks

| https://github.com/ooovector/qsweepy-plotting


To be able to use qsweepy you should do the following:

| 1) Install PostgreSQL https://www.postgresql.org/download/
| 2) Create a database

.. code-block:: sql

    create database qsweepy;
    CREATE USER qsweepy WITH PASSWORD 'qsweepy';
    grant all privileges on database qsweepy to qsweepy;