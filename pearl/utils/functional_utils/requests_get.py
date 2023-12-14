import requests


def requests_get(url: str) -> requests.Response:
    """
    Uses the appropriate requests.get version
    depending on whether one is running
    it inside Meta infrastructure or not
    """
    try:
        from pearl.utils.meta_only.meta_requests_get import meta_requests_get

        return meta_requests_get(url)
    except ImportError:
        # This means we are outside Meta infrastructure
        return requests.get(url)
