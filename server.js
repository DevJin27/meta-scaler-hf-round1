import { createApplication } from "./app/main.js";

const application = await createApplication();

await application.start();
