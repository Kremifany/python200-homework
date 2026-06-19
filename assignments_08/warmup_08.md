Part 1: Warmup -- Check for Understanding
Answer each question in your own words in warmup_08.md. 
A sentence or two is enough for most questions -- you are demonstrating that you understood the concept, not writing an essay. Try to do this without AI assistance.
Cloud Concepts
These questions are based on the Cloud Overview lesson.
Cloud Concepts Question 1
What is the core economic model of cloud computing, and how does it differ from owning your own servers?
The core economic model is that you paying per request not per volume you use and the system adjusting to your needs in memory and everything that needed for your project to run.It is scalable ,  secure and reliable
Cloud Concepts Question 2
What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.
When your project  i sscaling and you need let say more memory or computing power vertical scaling its when you jest adding more memory and proccessors to the same machine, Horizontal scaling when the load is distributed to many machines/sites and when it is needed more sites are available for your disposal with a load helper provided from cloud services
Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
Horisontal scaling will be a good match here because the load of growing amout of users will be distributed wisely among several machines and when the load is less the loader will reduce number of machines in usage
A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
Looks like the vertical scaling will fit this task better because its more or less one time growing need in memory and speed not always changing.
A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
horisontal will fit here better because the split across machines itsa a horisontal scaling
Cloud Concepts Question 3
Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each.
Gmail - SaaS - The servise provided not just a resourse but the whole app 
Azure Virtual Machines - IaaS - Infrastructure-level virtual machines you manage (OS, patches, runtime); you are responsible for configuring and maintaining the VM.
Azure App Service - PaaS - It's a platform that manages the servers for your app
AWS S3 (Simple Storage Service) - IaaS its an raw storage infrastructure
GitHub Codespaces - PaaS - its a platform to run dev env 
Snowflake - SaaS -  fully managed data platform , data warehouse
For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.
IaaS - Infrasturcure for a platform like data warehose like Azure VM.
 I am as a dev responsible for managing OS, patches, runtime, my app, my data
PaaS - Platform that uses the Infrastructure like Asure App Service.
 I am as a dev responsible for managing my app code and data.
SaaS - Software that you  jast login and  using not maintaning anything yourself uses Platform services like Gmail. 
I am as a dev responsible for managing my data and settings.


Cloud Concepts Question 4
What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?
A managed data platform like Snowflake or Databricks is a service that handles storing and analyzing big data for you. It runs on a cloud provider like Azure underneath, but it hides the setup and maintenance, so you don't have to build and configure servers, storage, and clusters yourself. Using Azure directly, you'd have to assemble and manage all those pieces on your own. With a managed platform you gain speed, simplicity, and automatic scaling without managing infrastructure. What you give up is control over the low-level setup, some flexibility, and you pay more for that convenience.

Cloud Concepts Question 5
The lesson names two situations where the cloud is probably not the right choice. What are they?
1.When the data fits on a single machine and you don't have heavy compute needs 
2.When you lose connectivity

Azure Basics

These questions are based on the Getting Started with Azure lesson.
Azure Basics Question 1
What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?
Azure subscription — This is the top-level account that billing and access are tied to.
Resource group — This is a folder inside a subscription that holds related resources (VMs, storage, etc.) for a single project or person

Azure Basics Question 2
Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?
What "ephemeral" means in practice — Cloud Shell runs in a small Linux container managed by Azure, and by default that container is ephemeral: every time you close the shell, all the files and directories you created get deleted. 
What the course setup uses to make it persistent — It connects Cloud Shell to a file share — a named storage folder in Azure (similar to a network drive), backed by the storage account your instructors already created in your resource group. Once it's mounted, your entire home directory (~) persists between sessions — SSH keys, scripts, config files, anything under ~.

Azure Basics Question 3
What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?
Your private key stays on your machine and is never shared. Your public key is the one uploaded to the remote systems you want to connect to. SSH checks that the two match to confirm your identity, so no password is ever sent. Sharing the public key is safe because it can only be used to verify you — it can't be used to log in by itself or to figure out your private key.


Azure Basics Question 4
Run the following command in Cloud Shell without the --output table flag:
az account show
Paste the output into your answer. Then describe in one sentence what changes when you add --output table.

answer: 
its a json format
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#fkreminsky@gmail.com",
    "type": "user"
  }
}

with --output table: more clean readable table 
EnvironmentName    HomeTenantId                          IsDefault    Name                       State    TenantId
-----------------  ------------------------------------  -----------  -------------------------  -------  ------------------------------------
AzureCloud         0f040ddd-301f-4665-8677-7b21f129d605  True         CTD Nonprofit Sponsorship  Enabled  0f040ddd-301f-4665-8677-7b21f129d605
