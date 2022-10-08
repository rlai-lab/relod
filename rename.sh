#!/bin/bash

git filter-branch --env-filter '
if [ "$GIT_AUTHOR_NAME" = "YufengYuan" ];
then
    GIT_AUTHOR_NAME="Yan Wang";
    GIT_AUTHOR_EMAIL="yan28@ualbeta.ca";
fi
if [ "$GIT_COMMITTER_NAME" = "YufengYuan" ];
then
    GIT_COMMITTER_NAME="Yan Wang";
    GIT_COMMITTER_EMAIL="yan28@ualbeta.ca";
fi
' -- --all