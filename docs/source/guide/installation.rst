Installation
==========================

Prerequisites
---------------

To be able to use qsweepy you need postgresql. You can either use a
remote database, or install a local server. For local server installation,

| 1) Install PostgreSQL https://www.postgresql.org/download/
| 2) Create a database and a user using the following SQL commands in the
SQL terminal window

.. code-block:: sql

    create database qsweepy;
    create user qsweepy with password 'qsweepy';
    grant all privileges on database qsweepy to qsweepy;

These scripts can be downloaded from

| https://github.com/ooovector/qtlab_replacement

| https://github.com/ooovector/qsweepy-notebooks

| https://github.com/ooovector/qsweepy-plotting


