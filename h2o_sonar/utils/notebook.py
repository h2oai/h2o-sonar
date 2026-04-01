# Copyright 2017-2026 H2O.ai, Inc. All rights reserved.
import base64
import os

from h2o_sonar.lib.api import commons


try:
    from IPython import display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def _get_js_download_func(func_name: str) -> str:
    return f"""function {func_name} (base64Data, filename) {{
    // Decode Base64 data to binary
    const binaryData = atob(base64Data);
    // Convert binary data to a Uint8Array
    const uint8Array = new Uint8Array(binaryData.length);
    for (let i = 0; i < binaryData.length; i++) {{
        uint8Array[i] = binaryData.charCodeAt(i);
    }}
    // Create a Blob from the Uint8Array
    const blob = new Blob([uint8Array], {{ type: "application/octet-stream" }});
    // Create a URL for the Blob
    const blobUrl = URL.createObjectURL(blob);
    // Create a link element to trigger the download
    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = filename; // Set the filename
    link.click();
    // Clean up: Revoke the Blob URL
    URL.revokeObjectURL(blobUrl);
}}"""


def _get_js_set_text_var(var_name: str, text: str) -> str:
    return f"var {var_name} = '{text}';"


def _get_js_set_b64_var(var_name: str, data: bytes) -> str:
    base64_data: str = base64.b64encode(data).decode()
    return _get_js_set_text_var(var_name, base64_data)


def _get_js_delete_symbol(symbol_name: str) -> str:
    return f"{symbol_name} = null;"


def _get_js_call_func(func_name: str, *var_names: str) -> str:
    return f"{func_name}({', '.join(var_names)});"


def _call_nb_display(
    script: str, before_msg_html: str = "", after_msg_html: str = ""
) -> None:
    if not HAS_IPYTHON:
        commons.raise_opt_import_err("ipython")

    if before_msg_html:
        display.display(display.HTML(before_msg_html))
    display.display(display.Javascript(script))
    # clear the Javascript for downloading, it's not persisted
    display.clear_output()
    if after_msg_html:
        display.display(display.HTML(after_msg_html))


def _determine_content(
    filename: str, content: bytes, binary: bool = True
) -> tuple[str, bytes]:
    if not content:
        with open(filename, "rb" if binary else "r") as fd:
            return os.path.basename(filename), fd.read()
    else:
        return filename, content


def _prep_download(filename: str, content: bytes = b"") -> str:
    download_func_name = "_h2o_sonar_nb_download"
    content_var_name = "_h2o_sonar_nb_content"
    filename_var_name = "_h2o_sonar_nb_filename"
    filename, raw_content = _determine_content(filename, content)
    call_str = _get_js_call_func(
        download_func_name, content_var_name, filename_var_name
    )
    script_list = [
        f"{_get_js_download_func(download_func_name)}",
        f"{_get_js_set_b64_var(content_var_name, raw_content)}",
        f"{_get_js_set_text_var(filename_var_name, filename)}",
        f"{call_str}\n",
        f"{_get_js_delete_symbol(content_var_name)}",
        f"{_get_js_delete_symbol(filename_var_name)}",
        f"{_get_js_delete_symbol(download_func_name)}",
    ]
    return "".join(script_list)


def download(filename: str) -> None:
    """Download a file. Used when files are not accessible outside the Jupyter
    Notebooks's kernel environment.

    Example usage:
    result.log(path="./dia-demo.log")
    download("./dia-demo.log")
    """
    script = _prep_download(filename)
    _call_nb_display(
        script, f"Preparing to download {filename}", f"Downloading {filename}"
    )
