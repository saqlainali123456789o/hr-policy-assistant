import streamlit as st

from pdf_processor import (
    extract_text_from_pdf,
    create_chunks
)

from rag_pipeline import HRPolicyRAG


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="wide"
)


st.title("🏢 HR Policy Assistant")

st.caption(
    "Ask questions about your uploaded HR policy document."
)


if "rag" not in st.session_state:

    st.session_state.rag = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 HR Policy")

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"]
    )

    if uploaded_file:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button(
            "🔍 Process Policy",
            use_container_width=True
        ):

            with st.spinner(
                "Processing HR policy..."
            ):

                pdf_bytes = uploaded_file.getvalue()

                pages = extract_text_from_pdf(
                    pdf_bytes
                )

                chunks = create_chunks(
                    pages
                )

                rag = HRPolicyRAG()

                rag.build_knowledge_base(
                    chunks
                )

                st.session_state.rag = rag

                st.session_state.messages = []

            st.success(
                f"Policy processed successfully!"
            )

            st.info(
                f"Pages: {len(pages)}\n\n"
                f"Chunks: {len(chunks)}"
            )


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# MAIN CHAT
# --------------------------------------------------

if st.session_state.rag is None:

    st.info(
        "👈 Upload an HR policy PDF and click "
        "'Process Policy' to start."
    )

else:

    st.success(
        "HR Policy is ready. Ask your question below."
    )


    # Display history

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    question = st.chat_input(
        "Ask a question about the HR policy..."
    )


    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):

            st.markdown(question)


        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the HR policy..."
            ):

                try:

                    answer, sources = (
                        st.session_state.rag
                        .generate_answer(question)
                    )

                    st.markdown(answer)

                    st.markdown(
                        "---"
                    )

                    st.markdown(
                        "**Retrieved policy sections:**"
                    )

                    for source in sources:

                        with st.expander(
                            f"📄 Page {source['page']} "
                            f"— similarity: "
                            f"{source['score']:.3f}"
                        ):

                            st.write(
                                source["text"]
                            )


                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )
