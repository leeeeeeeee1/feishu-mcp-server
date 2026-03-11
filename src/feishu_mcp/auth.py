"""Feishu authentication module."""

import os

import lark_oapi as lark


def create_client() -> lark.Client:
    """Create a Feishu API client with tenant token auth.

    Environment variables:
        FEISHU_APP_ID: App ID from Feishu Open Platform
        FEISHU_APP_SECRET: App Secret from Feishu Open Platform
        FEISHU_DOMAIN: Optional. Use "https://open.larksuite.com" for Lark international.
                       Defaults to "https://open.feishu.cn" (China).
    """
    app_id = os.environ.get("FEISHU_APP_ID")
    app_secret = os.environ.get("FEISHU_APP_SECRET")

    if not app_id or not app_secret:
        raise ValueError(
            "FEISHU_APP_ID and FEISHU_APP_SECRET environment variables are required"
        )

    domain = os.environ.get("FEISHU_DOMAIN", "")

    builder = lark.Client.builder().app_id(app_id).app_secret(app_secret)

    if domain and "larksuite" in domain:
        builder = builder.domain(lark.LARK_DOMAIN)
    else:
        builder = builder.domain(lark.FEISHU_DOMAIN)

    return builder.build()
