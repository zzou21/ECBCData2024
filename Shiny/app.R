library(shiny)
library(ggiraph)

# Source the module
source("modules/Virginia_Light_Dark.R")

# Load the data
df <- read.csv("../Projection/data/projectionResultWithMetaData_G1_VA.csv")

df_va <- df |>
  rename(
    "Filename" = `File.Name`,
    "P1" = `Projection..1`,
    "P2" = `Projection..2`,
    "P3" = `Projection..3`,
    "Title" = `Manuscript.Title`,
    "Year" = `Publication.Year`
  ) |>
  filter(P1 * P2 * P3 != 0)

# K-means clustering
set.seed(27708) # For reproducibility

# Standardization
df_k <- df_va |>
  mutate(
    P1_scaled = scale(P1),
    P2_scaled = scale(P2),
    P3_scaled = scale(P3)
  ) |>
  select(Filename, P1_scaled, P2_scaled, P3_scaled)

k_1 <- 3
k_2 <- 3
k_3 <- 4

kmeans_P1 <- kmeans(matrix(df_k$P1_scaled, ncol = 1), centers = k_1, nstart = 25)
kmeans_P2 <- kmeans(matrix(df_k$P2_scaled, ncol = 1), centers = k_2, nstart = 25)
kmeans_P3 <- kmeans(matrix(df_k$P3_scaled, ncol = 1), centers = k_3, nstart = 25)

# Add the cluster assignments to the original data frame
df_va <- df_va |> 
  mutate(cluster_P1 = kmeans_P1$cluster,
         cluster_P2 = kmeans_P2$cluster,
         cluster_P3 = kmeans_P3$cluster)

ui <- fluidPage(
  titlePanel("Interactive Clustering Plot"),
  sidebarLayout(
    sidebarPanel(
      # Add any sidebar inputs here
    ),
    mainPanel(
      plot_ui("plot1")
    )
  )
)

server <- function(input, output, session) {
  data <- reactive({
    df_va
  })
  
  plot_server("plot1", data)
}

shinyApp(ui, server)
