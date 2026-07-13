import RecentProjects from "~/home/RecentProjects";
import type { Route } from "./+types/home";
import UploadedDatasets from "~/home/UploadedDatasets";
import PrebuiltDatasets from "~/home/PrebuiltDatasets";
import SystemStatus from "~/components/SystemStatus";
import { useEffect, useState } from "react";
import { timeJoin } from "~/home/constants";
import LoginSignupModal from "~/Authenticate/LoginSignupModal";
import useModal from "~/customHooks/useModal";
import CreateProjectModal from "~/home/CreateProjectModal";

export function meta({}: Route.MetaArgs) {
  return [{ title: "The Observational Engine | Dashboard" }];
}

export default function Home() {
  const [session, setSession] = useState(0);
  const [datasetId, setDatasetId] = useState("");
  const { customModal, toggleModal } = useModal((closeHandler) => (
    <LoginSignupModal closeHandler={closeHandler}></LoginSignupModal>
  ));
  const closeHandler = function () {
    setDatasetId("");
  };
  const openHandler = function (datasetId: string) {
    projectModal.toggleModal(true);
    setDatasetId(datasetId);
  };
  const projectModal = useModal((closeModal) => (
    <CreateProjectModal
      datasetId={datasetId}
      closeHandler={() => {
        closeModal();
        closeHandler;
      }}
    ></CreateProjectModal>
  ));
  let mounted = true;
  useEffect(() => {
    const handler = function () {
      projectModal.toggleModal(false);
      closeHandler();
      toggleModal(true);
    };
    window.addEventListener("auth-required", handler);
    return () => window.removeEventListener("auth-required", handler);
  }, []);

  useEffect(() => {
    setTimeout(async function clock() {
      if (!mounted) {
        return;
      }
      setSession(new Date().getTime() - timeJoin);
      setTimeout(clock, 45 * 1000);
    }, 45 * 1000);
    return () => {
      mounted = false;
    };
  });
  return (
    <>
      <div className="grow bg-surface-dim p-8 overflow-y-auto">
        <section className="mb-12">
          <div className="flex items-baseline justify-between mb-6">
            <h1 className="text-2xl font-headline font-extrabold text-white tracking-tight">
              Observational Dashboard
            </h1>
            <span className="text-xs text-on-surface-variant font-medium">
              Session:{" "}
              {`${Math.round(session / 60 / 60 / 1000)}h ${Math.round(session / 60 / 1000)}m`}{" "}
              active
            </span>
          </div>
          <PrebuiltDatasets openHandler={openHandler}></PrebuiltDatasets>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <section className="lg:col-span-8 space-y-6">
            <RecentProjects></RecentProjects>
          </section>

          <section className="lg:col-span-4 space-y-6 pb-10">
            <UploadedDatasets toggleAuthen={toggleModal}></UploadedDatasets>
          </section>
        </div>
        <SystemStatus></SystemStatus>
        {customModal()}
        {projectModal.customModal()}
      </div>
    </>
  );
}
