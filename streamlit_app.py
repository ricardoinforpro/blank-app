# Add pandas excel support to imports
import streamlit as st
import pandas as pd
# import plotly.express as px # No longer used if charts are removed
from io import BytesIO
import logging # Import logging
from datetime import datetime # Import datetime
import numpy as np
import duckdb
from io import BytesIO

# --- Audit Log Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='audit.log', # Log file name
    filemode='a' # Append mode
)

def log_action(action_type, details=""):
    """Logs a user action to the audit file."""
    try:
        # Get user info if available (Streamlit doesn't provide direct user ID easily without authentication)
        # Using session ID as a proxy for a session/user
        session_id = st.runtime.scriptrunner.get_script_run_ctx().session_id
        log_message = f"Session: {session_id} | Action: {action_type} | Details: {details}"
        logging.info(log_message)
    except Exception as e:
        logging.error(f"Failed to log action: {e}") # Log errors during logging itself

# --- End Audit Log Configuration ---


# Configure page and caching settings
# Change these English texts to Portuguese:
st.set_page_config(
    page_title="Portal da Transparência - Remuneração",
    layout="wide",
    initial_sidebar_state="expanded"
)

#st.markdown("""
#    <style>
#        #MainMenu {visibility: hidden;}
#        footer {visibility: hidden;}
#        header {visibility: hidden;}
#    </style>
#""", unsafe_allow_html=True)

@st.cache_resource
def load_data_by_period(period_ano_mes):  # ex: '2025/03'
    periodo_int = int(period_ano_mes.replace('/', ''))  # 202503
    df = pd.read_parquet('remuneracao.parquet', engine='pyarrow', filters=[('periodo', '=', periodo_int)])

    # Tipagem leve
    df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int16')
    df['valor_bruto'] = pd.to_numeric(df['valor_bruto'], errors='coerce').astype('float32')
    df['valor_liquido'] = pd.to_numeric(df['valor_liquido'], errors='coerce').astype('float32')
    df['periodo'] = pd.to_numeric(df['periodo'], errors='coerce').astype('Int32')

    for col in ['nome_orgao_lotacao', 'nome_cargo', 'regime', 'tipo_cargo']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

@st.cache_data(show_spinner=True)
def load_periodos():
    df_periodos = pd.read_parquet('remuneracao.parquet', columns=['periodo'], engine='pyarrow')
    df_periodos = pd.to_numeric(df_periodos['periodo'], errors='coerce').dropna().astype('int32')
    datas_formatadas = pd.to_datetime(df_periodos.astype(str), format='%Y%m', errors='coerce').dt.strftime('%Y/%m')
    return sorted(datas_formatadas.dropna().unique().tolist(), reverse=True)

#@st.cache_data(show_spinner=True)(ttl=3600, show_spinner="Carregando dados...")
@st.cache_resource
def load_data():
    # Leitura otimizada do arquivo Parquet com dtype eficiente
    #df = pd.read_parquet('remuneracao.parquet')
    df = pd.read_parquet('remuneracao.parquet', engine='pyarrow')

    # Garantir que os tipos estejam otimizados (caso o Parquet não preserve tudo corretamente)
    df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int16')
    df['valor_bruto'] = df['valor_bruto'].astype('float32')
    df['valor_liquido'] = df['valor_liquido'].astype('float32')
    df['periodo'] = pd.to_numeric(df['periodo'], errors='coerce').astype('Int32')


    # Colunas categóricas (economia grande de RAM e melhoria em filtros)
    categorical_cols = ['nome_orgao_lotacao', 'nome_cargo', 'regime', 'tipo_cargo']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df

@st.cache_data(show_spinner="🔄 Carregando dados...")
def filter_dataframe_duckdb(df, global_search, selected_orgao, selected_cargo, selected_period, selected_ano, selected_regime, selected_tipo_cargo):
    filter_details = f"Global='{global_search}', Orgão='{selected_orgao}', Cargo='{selected_cargo}', Período='{selected_period}', Ano='{selected_ano}', Regime='{selected_regime}', TipoCargo='{selected_tipo_cargo}'"
    log_action("Filtro Aplicado", filter_details)

    df_filtered = df #.copy()

    # Constrói as cláusulas WHERE
    where_clauses = []

    if selected_orgao != 'Todos':
        where_clauses.append(f"nome_orgao_lotacao = '{selected_orgao}'")

    if selected_cargo != 'Todos':
        where_clauses.append(f"nome_cargo = '{selected_cargo}'")

    if selected_period != 'Todos':
        try:
            period_int = int(selected_period.replace('/', ''))
            where_clauses.append(f"periodo = {period_int}")
        except:
            st.error("Erro ao processar o período selecionado")

    if selected_ano != 'Todos':
        where_clauses.append(f"ano = {int(selected_ano)}")

    if selected_regime != 'Todos':
        where_clauses.append(f"regime = '{selected_regime}'")

    if selected_tipo_cargo != 'Todos':
        where_clauses.append(f"tipo_cargo = '{selected_tipo_cargo}'")

    if global_search:
        busca = global_search.lower().replace("'", "''")  # Escapar aspas
        search_condition = f"""(
            lower(nome_servidor) ILIKE '%{busca}%' OR
            lower(nome_cargo) ILIKE '%{busca}%' OR
            lower(nome_orgao_lotacao) ILIKE '%{busca}%' OR
            lower(regime) ILIKE '%{busca}%' OR
            lower(tipo_cargo) ILIKE '%{busca}%'
        )"""
        where_clauses.append(search_condition)

    where_clause = " AND ".join(where_clauses)
    query = f"SELECT * FROM df"
    if where_clause:
        query += f" WHERE {where_clause}"

    return duckdb.query(query).to_df()

@st.cache_data(show_spinner=True)
def get_unique_values(df, column):
    valores_unicos = df[column].dropna().unique().tolist()
    return ['Todos'] + sorted(valores_unicos)

@st.cache_data(show_spinner=True)
def filter_dataframe(df, global_search, selected_orgao, selected_cargo, selected_period, selected_ano, selected_regime, selected_tipo_cargo):
    # Log filter application attempt
    filter_details = f"Global='{global_search}', Orgão='{selected_orgao}', Cargo='{selected_cargo}', Período='{selected_period}', Ano='{selected_ano}', Regime='{selected_regime}', TipoCargo='{selected_tipo_cargo}'"
    log_action("Filtro Aplicado", filter_details)

    filtered_df = df #.copy()

    # Apply global search first across relevant text columns
    if global_search:
        search_cols = ['nome_servidor', 'nome_cargo', 'nome_orgao_lotacao', 'regime', 'tipo_cargo']
        search_cols = [col for col in search_cols if col in filtered_df.columns]
        
        # Convert categorical columns to string for searching
        search_df = pd.DataFrame()
        for col in search_cols:
            if col in filtered_df.columns:
                search_df[col] = filtered_df[col].astype(str)
        
        # Perform the search on string columns
        search_series = search_df.fillna('').agg(' '.join, axis=1)
        filtered_df = filtered_df[search_series.str.contains(global_search, case=False, na=False)]

    # Apply specific filters
    if selected_orgao != 'Todos':
        if 'nome_orgao_lotacao' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['nome_orgao_lotacao'] == selected_orgao]

    if selected_cargo != 'Todos':
        if 'nome_cargo' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['nome_cargo'] == selected_cargo]

    if selected_period != 'Todos':
        if 'periodo' in filtered_df.columns:
            try:
                period_int = int(selected_period.replace('/', ''))
                filtered_df = filtered_df[filtered_df['periodo'] == period_int]
            except ValueError:
                st.error(f"Formato de período inválido: {selected_period}")
            except KeyError:
                st.error("Coluna 'periodo' não encontrada.")

    if selected_ano != 'Todos':
        if 'ano' in filtered_df.columns:
            try:
                filtered_df = filtered_df[filtered_df['ano'] == int(selected_ano)]
            except ValueError:
                st.error(f"Formato de ano inválido: {selected_ano}")
            except KeyError:
                st.error("Coluna 'ano' não encontrada.")

    if selected_regime != 'Todos':
        if 'regime' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['regime'] == selected_regime]

    if selected_tipo_cargo != 'Todos':
        if 'tipo_cargo' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['tipo_cargo'] == selected_tipo_cargo]

    return filtered_df  # Always return the DataFrame

@st.cache_data
def get_cargos_por_orgao(df: pd.DataFrame, orgao: str) -> list:
    if orgao == 'Todos':
        return sorted(df['nome_cargo'].dropna().unique())
    return sorted(df.loc[df['nome_orgao_lotacao'] == orgao, 'nome_cargo'].dropna().unique())

@st.cache_data(show_spinner=True)
def get_period_values(df):
    periods = pd.to_datetime(df['periodo'], format='%Y%m', errors='coerce')
    formatted = periods.dropna().dt.strftime('%Y/%m')
    return sorted(datas_formatadas.dropna().unique().tolist(), reverse=True)

def main():
    # Initialize session state for pagination
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1
    
    if 'page_size' not in st.session_state:
        st.session_state.page_size = 50

    # Log application access
    log_action("Acesso", "Aplicação Iniciada")

    # Helper functions for BR number formatting
    def format_br_number(num, precision=0):
        return f"{num:_.{precision}f}".replace('.', ',').replace('_', '.')

    def format_br_currency(num):
        # Format as currency, then swap separators
        formatted = f"R$ {num:_.2f}".replace('.', ',').replace('_', '.')
        # Ensure R$ symbol is correctly placed
        parts = formatted.split(',')
        if len(parts) > 1:
            integer_part = parts[0]
            decimal_part = parts[1]
            # Remove potential extra R$ if number was negative
            if integer_part.count('R$') > 1:
                integer_part = integer_part.replace('R$','',1)
            # Handle negative sign placement
            if integer_part.startswith('-'):
                return f"-R$ {integer_part[1:]},{decimal_part}"
            else:
                return f"{integer_part},{decimal_part}"
        else:
            # Handle cases without decimals if necessary, though unlikely for currency
            if formatted.count('R$') > 1:
                formatted = formatted.replace('R$','',1)
            if formatted.startswith('-'):
                return f"-R$ {formatted[1:]}"
            else:
                return formatted

    st.title('Remuneração de Servidores')

    # --- Sidebar Filters ---
    st.sidebar.title('Filtros')
    
    # Add Clear Filters Button (except period)
    if st.sidebar.button("🧹 Limpar Filtros"):
        selected_orgao = 'Todos'
        selected_cargo = 'Todos'
        selected_ano = 'Todos'
        selected_regime = 'Todos'
        selected_tipo_cargo = 'Todos'
        st.rerun()

    # Initialize filters first
    all_periods = load_periodos()
    default_period = all_periods[-1] if all_periods else None
    selected_period = st.sidebar.selectbox("Período (AAAA/MM)", all_periods, index=0)
    
    df = load_data_by_period(selected_period)
    
    # Initialize other filters
    selected_ano = st.sidebar.selectbox('Ano', get_unique_values(df, 'ano'))
    selected_orgao = st.sidebar.selectbox("Órgão", ['Todos'] + list(sorted(df['nome_orgao_lotacao'].dropna().unique())))
    
    if selected_orgao != 'Todos':
        cargos_filtrados = get_cargos_por_orgao(df, selected_orgao)
        selected_cargo = st.sidebar.selectbox("Cargo", ['Todos'] + cargos_filtrados, key='cargo_filtered')
    else:
        cargos_filtrados = df['nome_cargo'].dropna().unique()
        selected_cargo = st.sidebar.selectbox("Cargo", ['Todos'] + list(sorted(cargos_filtrados)), key='cargo_all')
    
    selected_regime = st.sidebar.selectbox('Regime', get_unique_values(df, 'regime'))
    selected_tipo_cargo = st.sidebar.selectbox('Tipo Cargo', get_unique_values(df, 'tipo_cargo'))

    # Now display active filters after all filters are initialized
    active_filters = []
    if selected_period:
        active_filters.append(f"📅 Período: {selected_period}")
    if selected_orgao != 'Todos':
        active_filters.append(f"🏢 Órgão: {selected_orgao}")
    if selected_cargo != 'Todos':
        active_filters.append(f"👔 Cargo: {selected_cargo}")
    if selected_ano != 'Todos':
        active_filters.append(f"📆 Ano: {selected_ano}")
    if selected_regime != 'Todos':
        active_filters.append(f"📋 Regime: {selected_regime}")
    if selected_tipo_cargo != 'Todos':
        active_filters.append(f"🔖 Tipo Cargo: {selected_tipo_cargo}")

    if active_filters:
        st.write("**Filtros ativos:**", " | ".join(active_filters))
        st.divider()

    # --- Column Visibility ---
    st.sidebar.title('Colunas Visíveis')
    columns = {
        'Nome do Servidor': 'nome_servidor',
        'Nome do Cargo': 'nome_cargo',
        'Órgão': 'nome_orgao_lotacao',
        'Vencimentos': 'valor_bruto',
        'Valor Líquido': 'valor_liquido',
        'Mês': 'mes',
        'Ano': 'ano',
        'Regime': 'regime',
        'Tipo Cargo': 'tipo_cargo',
        'Período': 'periodo'
    }

    available_columns_dict = {k: v for k, v in columns.items() if v in df.columns}
    # Set default to all available columns
    visible_columns = st.sidebar.multiselect(
        'Selecione as colunas',
        list(available_columns_dict.keys()),
        default=list(available_columns_dict.keys()) # Default to all
    )

    
    # --- Metrics Display (with BR formatting) ---
    # Move global search before metrics
    global_search = st.text_input(
        label="🔎 Busca (nome, cargo, órgão, etc.)",
        placeholder="🔎 Digite nome, cargo, órgão, etc.",
        label_visibility="collapsed",
        key="global_search_input"  # Add unique key
    )
    
    # Apply filtering before metrics
    filtered_df = filter_dataframe(
        df, global_search, selected_orgao, selected_cargo,
        selected_period, selected_ano, selected_regime, selected_tipo_cargo
    )
    log_action("Dados", f"Filtrados {len(filtered_df)} registros")
    
    # Now we can display metrics with filtered data
    col1, col2, col3 = st.columns(3)
    total_servidores = filtered_df['nome_servidor'].nunique()
    valor_bruto_total = filtered_df['valor_bruto'].sum()
    valor_bruto_medio = filtered_df['valor_bruto'].mean()
    valor_bruto_mediana = filtered_df['valor_bruto'].median()

    with col1:
        st.metric(
            label="👥 Total de Servidores",
            value=f"{total_servidores:,}".replace(",", "."),
            help="Quantidade distinta de servidores ativos, aposentados e pensionistas no período selecionado"
        )
    with col2:
        st.metric("💰 Total de Vencimentos", format_br_currency(valor_bruto_total))
    with col3:
        st.metric("📈 Mediana dos Vencimentos", format_br_currency(valor_bruto_mediana))

    # Remove this duplicate section
    # global_search = st.text_input(
    #     label="🔎 Busca (nome, cargo, órgão, etc.)",
    #     placeholder="🔎 Digite nome, cargo, órgão, etc.",
    #     label_visibility="collapsed"
    # )

    # Remove this duplicate filtering section
    # filtered_df = filter_dataframe(
    #     df, global_search, selected_orgao, selected_cargo,
    #     selected_period, selected_ano, selected_regime, selected_tipo_cargo
    # )
    # log_action("Dados", f"Filtrados {len(filtered_df)} registros")

    # Helper function for BR number formatting
    def format_br_number(num, precision=0):
        return f"{num:_.{precision}f}".replace('.', ',').replace('_', '.')

    def format_br_currency(num):
         # Format as currency, then swap separators
        formatted = f"R$ {num:_.2f}".replace('.', ',').replace('_', '.')
        # Ensure R$ symbol is correctly placed
        parts = formatted.split(',')
        if len(parts) > 1:
            integer_part = parts[0]
            decimal_part = parts[1]
            # Remove potential extra R$ if number was negative
            if integer_part.count('R$') > 1:
                 integer_part = integer_part.replace('R$','',1)
            # Handle negative sign placement
            if integer_part.startswith('-'):
                 return f"-R$ {integer_part[1:]},{decimal_part}"
            else:
                 return f"{integer_part},{decimal_part}"
        else:
            # Handle cases without decimals if necessary, though unlikely for currency
            if formatted.count('R$') > 1:
                 formatted = formatted.replace('R$','',1)
            if formatted.startswith('-'):
                 return f"-R$ {formatted[1:]}"
            else:
                 return formatted


    # --- Tabelas Sintéticas (in Tabs) ---
    tab_names = []
    
    # Always show Regime summary
    tab_names.append("📊 Resumo por Regime")
    # Show Órgão summary only when no specific organ is selected
    if selected_orgao == 'Todos':
        tab_names.append("📊 Resumo por Órgão")
    # Show Cargo summary only when no specific cargo is selected
    if selected_cargo == 'Todos':
        tab_names.append("📊 Resumo por Cargo")
    
    # Create tabs if we have any tab names
    if tab_names:
        tabs = st.tabs(tab_names)

        # --- Resumo por Regime ---
        if "📊 Resumo por Regime" in tab_names:
            with tabs[tab_names.index("📊 Resumo por Regime")]:
                st.markdown("### Resumo por Regime")
                if not filtered_df.empty and 'regime' in filtered_df.columns:
                    regime_group = (
                        filtered_df
                        .groupby('regime', observed=True)
                        .agg(
                            Quantidade=('nome_servidor', 'nunique'),
                            Valor_Bruto_Total=('valor_bruto', 'sum'),
                            Valor_Bruto_Médio=('valor_bruto', 'mean')
                        )
                        .reset_index()
                        .sort_values('Valor_Bruto_Total', ascending=False)
                    )

                    # Cálculo da porcentagem
                    total_servidores = regime_group['Quantidade'].sum()
                    regime_group['% do Total'] = (regime_group['Quantidade'] / total_servidores * 100).round(1).astype(str) + '%'

                    regime_group = regime_group[['regime', 'Quantidade', '% do Total', 'Valor_Bruto_Total', 'Valor_Bruto_Médio']]

                    regime_group['Valor_Bruto_Médio'] = regime_group['Valor_Bruto_Médio'].apply(format_br_currency)
                    regime_group['Valor_Bruto_Total'] = regime_group['Valor_Bruto_Total'].apply(format_br_currency)

                    st.dataframe(regime_group, use_container_width=True, hide_index=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.button("🖨️ Imprimir", key='print-regime', on_click=lambda: st.markdown("<script>window.print()</script>", unsafe_allow_html=True))
                    with col2:
                        csv_data = regime_group.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📄 CSV",
                            csv_data,
                            "dados_regime.csv",
                            "text/csv",
                            key='export-csv-regime'
                        )
                    with col3:
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            regime_group.to_excel(writer, index=False, sheet_name='Dados')
                        st.download_button(
                            "📊 Excel",
                            excel_buffer.getvalue(),
                            "dados_regime.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='export-excel-regime'
                        )

    # Resumo por Órgão
    if selected_orgao == 'Todos' and "📊 Resumo por Órgão" in tab_names:
        with tabs[tab_names.index("📊 Resumo por Órgão")]:
            st.markdown("### Resumo por Órgão")
            if not filtered_df.empty and 'nome_orgao_lotacao' in filtered_df.columns:
                orgao_group = (
                    filtered_df
                    .groupby('nome_orgao_lotacao', observed=True)
                    .agg(
                        Quantidade=('nome_servidor', 'nunique'),
                        Valor_Bruto_Total=('valor_bruto', 'sum'),
                        Valor_Bruto_Médio=('valor_bruto', 'mean')
                    )
                    .reset_index()
                    .sort_values('Valor_Bruto_Total', ascending=False)
                )
                orgao_group['Valor_Bruto_Médio'] = orgao_group['Valor_Bruto_Médio'].apply(format_br_currency)
                orgao_group['Valor_Bruto_Total'] = orgao_group['Valor_Bruto_Total'].apply(format_br_currency)
                
                st.dataframe(orgao_group, use_container_width=True, hide_index=True)
                
                # Export Options for Orgão Table
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.button("🖨️ Imprimir", key='print-orgao', on_click=lambda: st.markdown("<script>window.print()</script>", unsafe_allow_html=True))
                with col2:
                    csv_data = orgao_group.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📄 CSV",
                        csv_data,
                        "dados_orgao.csv",
                        "text/csv",
                        key='export-csv-orgao'
                    )
                with col3:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        orgao_group.to_excel(writer, index=False, sheet_name='Dados')
                    st.download_button(
                        "📊 Excel",
                        excel_buffer.getvalue(),
                        "dados_orgao.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='export-excel-orgao'
                    )

    # Resumo por Cargo
    if selected_cargo == 'Todos' and "📊 Resumo por Cargo" in tab_names:
        with tabs[tab_names.index("📊 Resumo por Cargo")]:
            st.markdown("### Resumo por Cargo")
            if not filtered_df.empty and 'nome_cargo' in filtered_df.columns:
                cargo_group = (
                    filtered_df
                    .groupby('nome_cargo', observed=True)
                    .agg(
                        Quantidade=('nome_servidor', 'nunique'),
                        Valor_Bruto_Médio=('valor_bruto', 'mean'),
                        Valor_Bruto_Total=('valor_bruto', 'sum')
                    )
                    .reset_index()
                    .sort_values('Valor_Bruto_Total', ascending=False)
                )
                cargo_group['Valor_Bruto_Médio'] = cargo_group['Valor_Bruto_Médio'].apply(format_br_currency)
                cargo_group['Valor_Bruto_Total'] = cargo_group['Valor_Bruto_Total'].apply(format_br_currency)
                
                st.dataframe(cargo_group, use_container_width=True, hide_index=True)
                
                # Export Options for Cargo Table
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.button("🖨️ Imprimir", key='print-cargo', on_click=lambda: st.markdown("<script>window.print()</script>", unsafe_allow_html=True))
                with col2:
                    csv_data = cargo_group.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📄 CSV",
                        csv_data,
                        "dados_cargo.csv",
                        "text/csv",
                        key='export-csv-cargo'
                    )
                with col3:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        cargo_group.to_excel(writer, index=False, sheet_name='Dados')
                    st.download_button(
                        "📊 Excel",
                        excel_buffer.getvalue(),
                        "dados_cargo.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='export-excel-cargo'
                    )

    # --- Data Table Display ---
    st.subheader('Dados Detalhados')

    # Get visible columns and create display dataframe
    visible_column_ids = [available_columns_dict[col] for col in visible_columns]
    
    # Calculate pagination
    start_idx = (st.session_state.page_number - 1) * st.session_state.page_size
    end_idx = start_idx + st.session_state.page_size
    display_df = filtered_df.iloc[start_idx:min(end_idx, len(filtered_df))][visible_column_ids]

    # Format and display dataframe
    format_dict = {}
    if 'valor_bruto' in visible_column_ids:
        format_dict['valor_bruto'] = lambda x: format_br_currency(x).replace('R$ ','')
    if 'valor_liquido' in visible_column_ids:
        format_dict['valor_liquido'] = lambda x: format_br_currency(x).replace('R$ ','')

    # Display the table
    st.dataframe(
        display_df.style.format(format_dict, na_rep='-'),
        use_container_width=True
    )

    # Pagination controls below table
    col_size, col_nav = st.columns([1, 4])
    
    with col_size:
        new_page_size = st.selectbox(
            "Itens por página",
            options=[10, 25, 50, 100, 500],
            index=2,
            key='items_per_page'
        )

    # Reset page number when page size changes
        if new_page_size != st.session_state.page_size:
            st.session_state.page_size = new_page_size
            st.session_state.page_number = 1
            st.rerun()

    # Calculate total pages
    total_pages = len(filtered_df) // st.session_state.page_size + (1 if len(filtered_df) % st.session_state.page_size > 0 else 0)
    total_pages = max(1, total_pages)
    st.session_state.page_number = max(1, min(st.session_state.page_number, total_pages))

    # Pagination navigation
    with col_nav:
        col1_pag, col2_pag, col3_pag, col4_pag, col5_pag = st.columns([1.5, 1.5, 2, 1.5, 1.5])
        
        with col1_pag:
            if st.button("⏮️ Primeira", key='btn_first'):
                st.session_state.page_number = 1
                st.rerun()
    
    with col2_pag:
        if st.button("⏪ Anterior", key='btn_prev'):
            st.session_state.page_number = max(1, st.session_state.page_number - 1)
            st.rerun()
    
    with col3_pag:
        st.write(f"Página {st.session_state.page_number} de {total_pages}")
    
    with col4_pag:
        if st.button("Próxima ⏩", key='btn_next'):
            st.session_state.page_number = min(total_pages, st.session_state.page_number + 1)
            st.rerun()
    
    with col5_pag:
        if st.button("Última ⏭️", key='btn_last'):
            st.session_state.page_number = total_pages
            st.rerun()

    # Change export section:
    st.subheader('Exportar Dados')
    
    # Create data dictionary first
    descriptions_map = {
        'nome_servidor': 'Nome do Servidor',
        'nome_cargo': 'Cargo do Servidor',
        'nome_orgao_lotacao': 'Órgão de Lotação',
        'valor_bruto': 'Vencimentos da Remuneração (R$)', 
        'valor_liquido': 'Valor Líquido da Remuneração (R$)',
        'mes': 'Mês de Referência (Numérico)',
        'ano': 'Ano de Referência',
        'regime': 'Regime de Trabalho',
        'tipo_cargo': 'Tipo do Cargo',
        'periodo': 'Período de Referência (AAAAMM)'
    }

    dict_data = {
        'Coluna': list(available_columns_dict.keys()),
        'Nome Técnico': list(available_columns_dict.values()),
        'Descrição': [descriptions_map.get(tech_name, 'N/A') for tech_name in available_columns_dict.values()]
    }
    data_dict = pd.DataFrame(dict_data)
    
    # Export buttons for detailed table
    col_exp1, col_exp2, col_exp3, col_exp4, _ = st.columns([1, 1, 1, 1, 2])
    
    with col_exp1:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 CSV",
            csv_data,
            "dados_detalhados.csv",
            "text/csv",
            key='export-csv-details'
        )
    
    with col_exp2:
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Dados')
        st.download_button(
            "📊 Excel",
            excel_buffer.getvalue(),
            "dados_detalhados.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key='export-excel-details'
        )
    
    with col_exp3:
        dict_buffer = BytesIO()
        with pd.ExcelWriter(dict_buffer, engine='openpyxl') as writer:
            data_dict.to_excel(writer, index=False, sheet_name='Dicionário de Dados')
        st.download_button(
            "📖 Dicionário",
            dict_buffer.getvalue(),
            "dicionario_dados.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key='export-dict-details'  # Changed key to be unique
        )

    # Remove all duplicate dictionary creation and export sections below
    dict_buffer = BytesIO()
    # Use try-except for robustness during Excel writing
    try:
        with pd.ExcelWriter(dict_buffer, engine='openpyxl') as writer:
            data_dict.to_excel(writer, index=False, sheet_name='Dicionário de Dados')
        st.download_button(
            "📖 Dicionário de Dados",
            dict_buffer.getvalue(),
            "dicionario_dados.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key='export-dict' # Added key
        )
    except Exception as e:
        st.error(f"Erro ao gerar dicionário: {e}")

    with col2:
        # Print version
        # Add logging for print action
        if st.button("🖨️ Versão para Impressão"):
            log_action("Exportação", "Versão para Impressão Solicitada")
            st.markdown(
                f"<script>window.print()</script>",
                unsafe_allow_html=True
            )

    with col3:
        # CSV Export
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Exportar CSV",
            csv_data,
            "dados_remuneracao.csv",
            "text/csv",
            key='export-csv',
            # Add logging on click
            on_click=log_action, args=("Exportação", "CSV Baixado")
        )

    with col4:
        # Excel Export
        excel_buffer = BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Export only visible columns to Excel to match user view
                filtered_df[visible_column_ids].to_excel(writer, index=False, sheet_name='Dados')
            st.download_button(
                "📊 Exportar Excel",
                excel_buffer.getvalue(),
                "dados_remuneracao.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='export-excel',
                # Add logging on click
                on_click=log_action, args=("Exportação", "Excel Baixado")
            )
        except Exception as e:
            st.error(f"Erro ao gerar Excel: {e}")

    # Remove the old download button
    # Optimized download functionality
    # Remove the second (duplicate) "Preparar Download" button section
    # if st.button("Preparar Download"):
    #     csv = filtered_df.to_csv(index=False).encode('utf-8')
    #     st.download_button(
    #         "Download dados filtrados (CSV)",
    #         csv,
    #         "dados_remuneracao.csv",
    #         "text/csv",
    #         key='download-csv' # This key conflicts with the other download button
    #     )

if __name__ == "__main__":
    main()
